
// monojni — the .NET host, as a library loaded into a process that has a JVM.
//
// This began as monohost, an ordinary executable, and the reason it is not one
// any more is worth writing down.
//
// The launcher runs three .NET programs: il2cpp.dll, Roslyn's csc.dll, and the
// bundle surgery tool. Android has no .NET, and carrying Debian's glibc beside
// Microsoft's linux-arm64 runtime -- the obvious answer, and what this used to
// do -- cannot work on a current Android: every app runs under a seccomp
// filter generated from the syscalls bionic itself makes, and it answers
// anything else with SIGSYS rather than ENOSYS, so a modern glibc dies
// registering rseq before reaching main() and prints nothing.
//
// Microsoft's Mono build for android-arm64 has no such problem; it is bionic's
// syscalls all the way down. But it is built for Android in a stronger sense
// than "compiled against bionic": .NET on Android implements
// System.Security.Cryptography by calling Java's javax.crypto through JNI.
// libSystem.Security.Cryptography.Native.Android.so receives its JavaVM in
// JNI_OnLoad, which only runs when Java's System.loadLibrary loads it. In a
// plain exec'd process there is no JavaVM, nothing calls JNI_OnLoad, and the
// first hash lands on a null pointer -- which is where a subprocess host died,
// inside SHA256.Create, on the checksum Roslyn takes of every source file.
//
// So the runtime has to live in a process that has a JVM, which on Android
// means an app process. It gets one of its own -- see MonoService and the two
// builder processes in the manifest -- rather than the launcher's, because
// the isolation a subprocess gave away for free is worth keeping:
//
//   il2cpp calls Environment.Exit, which ends the process it is running in.
//   In a builder that is a process whose whole purpose is to be ended.
//
//   il2cpp resolves parts of its own installation relative to the working
//   directory, and chdir() is process-wide.
//
//   The runtime cannot be initialised twice in a process, and there are four
//   programs to run. The builder is killed after each one, so each starts
//   clean.
//
// What is left here is the part hostfxr and hostpolicy would do on a desktop:
// find the runtime, decide which assemblies it may load, and start it. There
// is no hostfxr for Android to defer to -- Microsoft ships no apphost for the
// platform, and the one host that does exist, libnet-android.so, is a JNI
// library with no main() and no way to be told to run an arbitrary assembly.

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <jni.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define LOG_TAG "monojni"
#include <android/log.h>
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

typedef int (*coreclr_initialize_fn)(
    const char *exePath, const char *appDomainFriendlyName, int propertyCount,
    const char **propertyKeys, const char **propertyValues, void **hostHandle,
    unsigned int *domainId);

typedef int (*coreclr_execute_assembly_fn)(
    void *hostHandle, unsigned int domainId, int argc, const char **argv,
    const char *managedAssemblyPath, unsigned int *exitCode);

typedef int (*coreclr_shutdown_2_fn)(
    void *hostHandle, unsigned int domainId, int *latchedExitCode);

// A growable string, for the trusted assembly list: every assembly the runtime
// is allowed to load, ':'-separated, which for a whole class library is tens of
// kilobytes -- too long for a fixed buffer and too important to truncate.
typedef struct { char *data; size_t len, cap; } strbuf;

// Where the exit code is left for the launcher, and how it gets written even
// when nothing here is given the chance to write it.
//
// Environment.Exit does not return. Mono implements it by recording the code
// and calling exit(), which unwinds the process from inside
// coreclr_execute_assembly -- so the Kotlin that started this run never gets
// control back, and the finally block that would have written the result file
// never runs. il2cpp exits that way, and a conversion that ended perfectly
// well would otherwise be indistinguishable from one the low memory killer
// took.
//
// atexit handlers do run for exit(), so the file is written from one. The
// code comes from the runtime rather than from here: Environment.Exit went
// through mono_environment_exitcode_set on its way past, and that value is
// the one the program meant to return.
static char g_result_path[PATH_MAX];
static unsigned int g_exit_code = 120;
static int g_have_exit_code = 0;
static int g_result_written = 0;
static int (*g_exitcode_get)(void) = NULL;

static void write_result(void) {
    if (g_result_written || g_result_path[0] == '\0') return;
    g_result_written = 1;

    // Before the result file, never after. The launcher stops reading the
    // output the pass after this file appears, so anything still sitting in
    // stdio's buffer at that moment is never read and is deleted with the
    // file. On the exit() path that is all of it: bionic runs atexit
    // handlers first and flushes the streams afterwards, so a program that
    // ends through Environment.Exit -- which is how il2cpp ends, including
    // when it ends badly -- had its entire output thrown away. That is why a
    // 375-second conversion could leave a convert.log of nothing at all.
    fflush(NULL);

    unsigned int code = g_exit_code;
    // A normal return has already put the real answer in g_exit_code; only
    // ask the runtime when this is the exit() path.
    if (!g_have_exit_code && g_exitcode_get) code = (unsigned int)g_exitcode_get();

    // Diagnostic: write the final code to convert.log before the result file
    // appears and the launcher stops reading the output.
    fprintf(stderr, "monojni: final exit code=%u\n", code);
    fflush(stderr);

    // Written to one side and renamed into place. The launcher treats the
    // result file existing as "the run is over and the code is readable", and
    // open-then-write leaves a moment where it exists and is empty; a seek to
    // a byte offset lands in the middle of one.
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s.tmp", g_result_path);
    FILE *f = fopen(tmp, "w");
    if (!f) return;
    fprintf(f, "%u", code);
    fflush(f);
    fclose(f);
    rename(tmp, g_result_path);
}

// Diagnostic: coreclr_shutdown_2 is given 20 seconds. If it hangs, the process
// ends here instead of leaving the launcher waiting for ever for a result
// file that would never appear. write() and _exit() are async-signal-safe.
static void shutdown_watchdog(int sig) {
    (void)sig;
    static const char m[] = "monojni: coreclr_shutdown_2 did not finish in 20s\n";
    write(STDERR_FILENO, m, sizeof(m) - 1);
    _exit(121);
}

// Diagnostic: a sampler that reports, every few seconds while the program runs,
// what this process is doing. It makes no assumption about the runtime: it only
// reads /proc/self. CPU time close to wall time means the program is working;
// CPU time far below it means it is waiting for something. The open files show
// what it is reading and writing.
static volatile int g_sampler_stop = 0;

static long sampler_status_kb(const char *key) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    size_t n = strlen(key);
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, n) == 0) {
            v = strtol(line + n, NULL, 10);
            break;
        }
    }
    fclose(f);
    return v;
}

// Fills the user and system CPU time of a /proc stat file, in clock ticks, and
// the state letter. Fields are read after the last ')' because the process
// name in field 2 may itself contain spaces or parentheses.
static int sampler_read_stat(const char *path, unsigned long *ut, unsigned long *st, char *state) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[1024];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    char *p = strrchr(buf, ')');
    if (!p) return 0;
    char s = '?';
    unsigned long u = 0, k = 0;
    if (sscanf(p + 1, " %c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %lu %lu", &s, &u, &k) < 3) return 0;
    if (ut) *ut = u;
    if (st) *st = k;
    if (state) *state = s;
    return 1;
}

static void sampler_report(const char *tag, int t) {
    long clk = sysconf(_SC_CLK_TCK);
    if (clk <= 0) clk = 100;
    unsigned long ut = 0, st = 0;
    sampler_read_stat("/proc/self/stat", &ut, &st, NULL);
    int r = 0, s = 0, d = 0, other = 0, total = 0;
    DIR *dir = opendir("/proc/self/task");
    if (dir) {
        struct dirent *e;
        while ((e = readdir(dir)) != NULL) {
            if (e->d_name[0] == '.') continue;
            char path[320];
            snprintf(path, sizeof path, "/proc/self/task/%s/stat", e->d_name);
            char state = '?';
            if (!sampler_read_stat(path, NULL, NULL, &state)) continue;
            total++;
            if (state == 'R') r++;
            else if (state == 'S') s++;
            else if (state == 'D') d++;
            else other++;
        }
        closedir(dir);
    }
    fprintf(stderr,
            "monojni: %s t=%ds rss=%ldMB peak=%ldMB cpu=%.1fs (user %.1f sys %.1f) "
            "threads=%d (running %d, sleeping %d, disk-wait %d, other %d)\n",
            tag, t, sampler_status_kb("VmRSS:") / 1024, sampler_status_kb("VmHWM:") / 1024,
            (double)(ut + st) / (double)clk, (double)ut / (double)clk, (double)st / (double)clk,
            total, r, s, d, other);
    fflush(stderr);
}

// Lists the regular files the process has open, so the log shows what it is
// reading. Sockets, pipes and anonymous handles are only counted.
static void sampler_fds(const char *tag, int t) {
    DIR *dir = opendir("/proc/self/fd");
    if (!dir) return;
    int shown = 0, other = 0;
    struct dirent *e;
    while ((e = readdir(dir)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char path[320], target[PATH_MAX];
        snprintf(path, sizeof path, "/proc/self/fd/%s", e->d_name);
        ssize_t n = readlink(path, target, sizeof target - 1);
        if (n <= 0) continue;
        target[n] = 0;
        if (target[0] != '/' || strncmp(target, "/dev/", 5) == 0 || strncmp(target, "/proc/", 6) == 0) {
            other++;
            continue;
        }
        if (shown < 40) {
            fprintf(stderr, "monojni: %s t=%ds fd %s -> %.200s\n", tag, t, e->d_name, target);
            shown++;
        } else {
            other++;
        }
    }
    closedir(dir);
    fprintf(stderr, "monojni: %s t=%ds open files shown=%d, others=%d\n", tag, t, shown, other);
    fflush(stderr);
}

static void *sampler_main(void *arg) {
    (void)arg;
    int t = 0;
    while (!g_sampler_stop) {
        sleep(1);
        t++;
        if (g_sampler_stop) break;
        if (t % 5 != 0) continue;
        sampler_report("sample", t);
        if (t == 15 || t == 45 || t == 75) sampler_fds("sample", t);
    }
    return NULL;
}

static int sb_add(strbuf *b, const char *s) {
    size_t n = strlen(s);
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 8192;
        while (cap < b->len + n + 1) cap *= 2;
        char *p = realloc(b->data, cap);
        if (!p) return 0;
        b->data = p;
        b->cap = cap;
    }
    memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
    return 1;
}

/**
 * Appends every .dll in [dir] to the trusted assembly list.
 *
 * Only .dll: the runtime pack keeps native libraries in the same directory as
 * System.Private.CoreLib, and offering it a .so as an assembly is not
 * something it recovers from. Duplicates are harmless -- the first occurrence
 * of a simple name wins, which is why the class library is added before the
 * program's own directory.
 */
static void add_dlls(strbuf *tpa, const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        const char *dot = strrchr(e->d_name, '.');
        if (!dot || strcmp(dot, ".dll") != 0) continue;
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        if (tpa->len) sb_add(tpa, ":");
        sb_add(tpa, path);
    }
    closedir(d);
}

static char *dup_jstring(JNIEnv *env, jstring s) {
    if (!s) return NULL;
    const char *c = (*env)->GetStringUTFChars(env, s, NULL);
    char *out = c ? strdup(c) : NULL;
    if (c) (*env)->ReleaseStringUTFChars(env, s, c);
    return out;
}

JNIEXPORT jint JNICALL
Java_dev_silksong_launcher_MonoBridge_nativeRun(
    JNIEnv *env, jclass clazz, jstring j_assembly, jobjectArray j_args,
    jstring j_cwd, jstring j_bcl, jstring j_runtime, jobjectArray j_env,
    jstring j_out, jstring j_result) {
    (void)clazz;

    // Diagnostic: confirm that nativeRun was entered.
    LOGI("nativeRun: entered");

    char *assembly = dup_jstring(env, j_assembly);
    char *cwd = dup_jstring(env, j_cwd);
    char *bcl = dup_jstring(env, j_bcl);
    char *runtime_dir = dup_jstring(env, j_runtime);
    char *out_path = dup_jstring(env, j_out);
    char *result_path = dup_jstring(env, j_result);

    // Registered before anything can fail, so that every path out of this
    // function -- including the ones that do not come back through it --
    // leaves the launcher an answer rather than a process that vanished.
    if (result_path) {
        snprintf(g_result_path, sizeof(g_result_path), "%s", result_path);
        atexit(write_result);
    }

    // Everything the runtime writes to stdout and stderr goes to a file the
    // launcher is tailing. A pipe would be the obvious choice and was the
    // first one; it cannot be used, because the descriptor has to reach this
    // process through the Intent that starts the service and the activity
    // manager refuses to carry file descriptors ("Not allowed to write file
    // descriptors here"). A path crosses that boundary as a string, and both
    // processes are the same application, so they share the directory.
    //
    // Redirected here rather than in the caller because the runtime captures
    // both descriptors when it starts and never consults them again.
    if (out_path) {
        int fd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
            if (fd > STDERR_FILENO) close(fd);
            // Line buffered, so the launcher sees progress rather than one
            // block of output when the program finally exits.
            setvbuf(stdout, NULL, _IOLBF, 0);
            setvbuf(stderr, NULL, _IONBF, 0);
        }
    }

    // Diagnostic: record entry and the JNI arguments in convert.log.
    fprintf(stderr,
            "monojni: nativeRun reached; assembly=%s bcl=%s runtime=%s\n",
            assembly ? assembly : "(null)",
            bcl ? bcl : "(null)",
            runtime_dir ? runtime_dir : "(null)");
    fflush(stderr);

    // Check required arguments only after stderr has been redirected, so
    // early failures are visible in convert.log too.
    if (!assembly || !bcl || !runtime_dir) {
        fprintf(stderr, "monojni: missing required JNI argument\n");
        fflush(stderr);
        return 120;
    }

    if (j_env) {
        jsize n = (*env)->GetArrayLength(env, j_env);
        for (jsize i = 0; i + 1 < n; i += 2) {
            char *k = dup_jstring(env, (jstring)(*env)->GetObjectArrayElement(env, j_env, i));
            char *v = dup_jstring(env, (jstring)(*env)->GetObjectArrayElement(env, j_env, i + 1));
            if (k && v) setenv(k, v, 1);
            free(k);
            free(v);
        }
    }

    // il2cpp resolves parts of its own installation relative to here.
    if (cwd && chdir(cwd) != 0) {
        fprintf(stderr, "monojni: cannot enter %s\n", cwd);
        return 120;
    }

    char app_dir[PATH_MAX];
    snprintf(app_dir, sizeof(app_dir), "%s", assembly);
    char *slash = strrchr(app_dir, '/');
    if (slash) *slash = '\0';

    strbuf tpa = {0};
    // The class library first, and the program's own directory second.
    //
    // The opposite is the usual convention -- an application that ships a
    // copy of a framework assembly generally means the one it was built
    // against -- and it is wrong here. il2cpp carries the Linux flavour of
    // System.Security.Cryptography, whose Interop.Crypto P/Invokes
    // libSystem.Security.Cryptography.Native.OpenSsl. That library does not
    // exist on Android and is not what the runtime pack ships; letting it
    // win produced a DllNotFoundException the moment Mono.Cecil hashed a
    // public key, which it does for the first assembly it reads.
    //
    // The class library shipped with the runtime is the one built for this
    // platform, and it is the one the native libraries beside it answer to.
    // Assemblies that are genuinely the program's own -- Mono.Cecil,
    // Unity.IL2CPP.* and the rest -- are not in the class library, so they
    // still resolve from the program's directory.
    add_dlls(&tpa, bcl);
    add_dlls(&tpa, app_dir);
    if (!tpa.len) {
        fprintf(stderr, "monojni: no managed assemblies in %s or %s\n", app_dir, bcl);
        return 120;
    }

    // By absolute path: this is loaded into an app process whose linker
    // namespace does search nativeLibraryDir, but the runtime is named
    // explicitly anyway so that the copy beside this library is the one that
    // answers, whatever else is loaded.
    char so_path[PATH_MAX];
    snprintf(so_path, sizeof(so_path), "%s/libmonosgen-2.0.so", runtime_dir);
    void *rt = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
    if (!rt) {
        fprintf(stderr, "monojni: cannot load %s: %s\n", so_path, dlerror());
        return 120;
    }

    coreclr_initialize_fn cc_init = (coreclr_initialize_fn)dlsym(rt, "coreclr_initialize");
    coreclr_execute_assembly_fn cc_exec =
        (coreclr_execute_assembly_fn)dlsym(rt, "coreclr_execute_assembly");
    if (!cc_init || !cc_exec) {
        fprintf(stderr, "monojni: the runtime does not export the CoreCLR hosting API\n");
        return 120;
    }

    // Mono uses SIGSEGV as a working part of the runtime -- null checks and
    // the collector's write barrier fault on a guard page and recover in the
    // handler -- so its handlers have to coexist with the ones Android's own
    // runtime installs in every process. Without chaining the two fight over
    // the same signals, an ordinary internal fault is reported as a crash,
    // and the crash reporter faults in turn: "Exiting early due to double
    // fault", with no stack trace to show for it.
    void (*set_signal_chaining)(int) = (void (*)(int))dlsym(rt, "mono_set_signal_chaining");
    void (*set_crash_chaining)(int) = (void (*)(int))dlsym(rt, "mono_set_crash_chaining");
    if (set_signal_chaining) set_signal_chaining(1);
    if (set_crash_chaining) set_crash_chaining(1);

    // Where the atexit handler reads the code from when the program leaves
    // through Environment.Exit rather than by returning.
    g_exitcode_get = (int (*)(void))dlsym(rt, "mono_environment_exitcode_get");

    // Invariant globalization because the runtime pack carries no ICU for
    // Android, and il2cpp's own runtimeconfig asks for it regardless.
    //
    // Server GC off explicitly. Roslyn's csc.runtimeconfig.json asks for it,
    // and a real host would honour that -- but this runtime collects with
    // SGen, which has no server collector, and nothing here reads that file
    // anyway. Saying false leaves no room for it to arrive by another route.
    const char *keys[] = {
        "TRUSTED_PLATFORM_ASSEMBLIES",
        "APP_PATHS",
        "APP_NI_PATHS",
        "APP_CONTEXT_BASE_DIRECTORY",
        "NATIVE_DLL_SEARCH_DIRECTORIES",
        "System.Globalization.Invariant",
        "System.GC.Server",
        "System.GC.Concurrent",
    };
    const char *values[] = {
        tpa.data, app_dir, app_dir, app_dir, runtime_dir, "true", "false", "false",
    };

    void *host = NULL;
    unsigned int domain = 0;

    // Diagnostic: show when Mono CoreCLR initialization begins.
    LOGI("nativeRun: starting CoreCLR");

    int hr = cc_init(assembly, "silksong-builder",
                     (int)(sizeof(keys) / sizeof(keys[0])), keys, values, &host, &domain);

    // Diagnostic: record CoreCLR initialization in convert.log.
    fprintf(stderr, "monojni: CoreCLR init hr=0x%08x\n", (unsigned)hr);
    fflush(stderr);

    // Diagnostic: record the CoreCLR initialization result in Logcat too.
    LOGI("nativeRun: CoreCLR result=0x%08x", (unsigned)hr);

    if (hr < 0) {
        fprintf(stderr, "monojni: the runtime failed to start: 0x%08x\n", (unsigned)hr);
        LOGE("coreclr_initialize failed: 0x%08x", (unsigned)hr);
        return 120;
    }

    int argc = j_args ? (int)(*env)->GetArrayLength(env, j_args) : 0;
    const char **argv = argc ? calloc((size_t)argc, sizeof(char *)) : NULL;
    for (int i = 0; i < argc; i++) {
        argv[i] = dup_jstring(env, (jstring)(*env)->GetObjectArrayElement(env, j_args, i));
    }

    LOGI("running %s with %d argument(s)", assembly, argc);
    unsigned int exit_code = 0;

    // Diagnostic: start the sampler just before the program does.
    {
        pthread_t sampler;
        if (pthread_create(&sampler, NULL, sampler_main, NULL) == 0) {
            pthread_detach(sampler);
        } else {
            fprintf(stderr, "monojni: could not start the sampler thread\n");
        }
    }
    hr = cc_exec(host, domain, argc, argv, assembly, &exit_code);
    g_sampler_stop = 1;
    sampler_report("on return", -1);
    sampler_fds("on return", -1);

    // Diagnostic: record assembly execution in convert.log.
    fprintf(stderr, "monojni: execute hr=0x%08x exit=%u\n",
            (unsigned)hr, exit_code);
    fflush(stderr);

    // Diagnostic: record the assembly execution result in Logcat too.
    LOGI("nativeRun: execute result=0x%08x exit=%u", (unsigned)hr, exit_code);

    if (hr < 0) {
        fprintf(stderr, "monojni: could not run %s: 0x%08x\n", assembly, (unsigned)hr);
        return 120;
    }

    // The program returned rather than exiting, so this is the real code and
    // the runtime's own copy should not be consulted.
    g_exit_code = exit_code;
    g_have_exit_code = 1;

    // Diagnostic (the one change in this build): shut the runtime down so the
    // managed side runs its ProcessExit handlers and flushes any output it is
    // still holding. Before this, a program that returned from Main instead of
    // calling Environment.Exit could lose whatever it had buffered, which
    // would look exactly like a program that printed nothing.
    coreclr_shutdown_2_fn cc_shutdown =
        (coreclr_shutdown_2_fn)dlsym(rt, "coreclr_shutdown_2");
    if (cc_shutdown) {
        fprintf(stderr, "monojni: shutting the runtime down to flush managed output\n");
        fflush(stderr);
        signal(SIGALRM, shutdown_watchdog);
        alarm(20);
        int latched = 0;
        int shr = cc_shutdown(host, domain, &latched);
        alarm(0);
        fprintf(stderr, "monojni: shutdown hr=0x%08x latched exit=%d\n",
                (unsigned)shr, latched);
        fflush(stderr);
    } else {
        fprintf(stderr, "monojni: coreclr_shutdown_2 is not exported by this runtime\n");
        fflush(stderr);
    }
    fflush(stdout);
    fflush(stderr);
    write_result();
    return (jint)exit_code;
}
