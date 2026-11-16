#!/usr/bin/env python3
"""Aggregate mask_const c312 TB2.1 mirror run (0.41 / 73 pass of 178) by hand-labelled domain.

Usage: python3 domain_stats.py traj_deep.json
Each of the 89 tasks is mapped to (domain, primary_skill_family). Output:
  - per-domain pass@all (sum of scores / (2*n_env)), env-level both-pass / split / both-fail
  - behavioural contrast PASS vs failure (steps, think, compactions, repeat_ratio)
"""
import json, collections, sys, statistics

D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "traj_deep.json"))

# (domain, skill) per env. skill in {tooling, debug-fix, data-sci, algo, security, ml-torch,
#  native-build, media-image, sysadmin-net, codegen-spec, reverse-eng}
T = {
 "build-pmars":                 ("软件构建/打包", "native-build"),
 "build-cython-ext":            ("软件构建/打包", "native-build"),
 "build-pov-ray":               ("软件构建/打包", "native-build,legacy"),
 "compile-compcert":            ("软件构建/打包", "native-build"),
 "sqlite-with-gcov":            ("软件构建/打包", "native-build"),
 "caffe-cifar-10":              ("软件构建/打包", "native-build,ml"),
 "rstan-to-pystan":             ("软件构建/打包", "stats-api-port"),
 "modernize-scientific-stack":  ("代码迁移/现代化", "tooling"),
 "cobol-modernization":         ("代码迁移/现代化", "tooling,logic"),
 "fix-git":                     ("git/vcs 操作", "tooling"),
 "git-leak-recovery":           ("git/vcs 操作", "tooling"),
 "git-multibranch":             ("git/vcs 操作", "sysadmin"),
 "sanitize-git-repo":           ("git/vcs 操作", "tooling,security"),
 "merge-diff-arc-agi-task":     ("git/vcs 操作", "tooling"),
 "configure-git-webserver":     ("服务器配置/网络", "sysadmin"),
 "nginx-request-logging":       ("服务器配置/网络", "sysadmin"),
 "pypi-server":                 ("服务器配置/网络", "sysadmin"),
 "openssl-selfsigned-cert":     ("服务器配置/网络", "sysadmin"),
 "mailman":                     ("服务器配置/网络", "sysadmin"),
 "kv-store-grpc":               ("服务器配置/网络", "codegen-spec"),
 "hf-model-inference":          ("服务器配置/网络", "sysadmin,ml"),
 "fix-code-vulnerability":      ("安全/vuln 修复", "security"),
 "vulnerable-secret":           ("安全/vuln 修复", "security,reveng"),
 "filter-js-from-html":         ("安全/vuln 修复", "security,spec"),
 "break-filter-js-from-html":   ("安全/vuln 修复", "security,adversarial"),
 "model-extraction-relu-logits":("安全/vuln 修复", "algo,adversarial"),
 "password-recovery":           ("取证/恢复", "reveng,tooling"),
 "db-wal-recovery":             ("取证/恢复", "reveng,db"),
 "sqlite-db-truncate":          ("取证/恢复", "reveng,db"),
 "crack-7z-hash":               ("取证/恢复", "tooling,crypto"),
 "db-wal":                      ("取证/恢复", "reveng,db"),
 "extract-elf":                 ("取证/恢复", "reveng"),
 "path-tracing-reverse":        ("取证/恢复", "reveng"),
 "gcode-to-text":               ("领域文件解析", "spec,vision"),
 "extract-moves-from-video":    ("领域文件解析", "vision"),
 "code-from-image":             ("领域文件解析", "vision"),
 "chess-best-move":             ("领域文件解析", "vision,reason"),
 "financial-document-processor":("领域文件解析", "vision,spec"),
 "count-dataset-tokens":        ("数据/科学计算", "data-sci"),
 "multi-source-data-merger":    ("数据/科学计算", "data-sci"),
 "reshard-c4-data":             ("数据/科学计算", "data-sci"),
 "train-fasttext":              ("数据/科学计算", "ml"),
 "mteb-retrieve":               ("数据/科学计算", "ml,api"),
 "mteb-leaderboard":            ("数据/科学计算", "ml,api"),
 "sam-cell-seg":                ("数据/科学计算", "ml,vision"),
 "bn-fit-modify":               ("数据/科学计算", "stats"),
 "mcmc-sampling-stan":          ("数据/科学计算", "stats,ml"),
 "raman-fitting":               ("数据/科学计算", "stats,domain"),
 "protein-assembly":            ("领域知识(生物)", "domain"),
 "dna-assembly":                ("领域知识(生物)", "domain"),
 "dna-insert":                  ("领域知识(生物)", "domain"),
 "sparql-university":           ("查询/KG 语言", "spec,db"),
 "query-optimize":              ("查询/KG 语言", "db,perf"),
 "log-summary-date-ranges":     ("查询/KG 语言", "spec"),
 "regex-log":                   ("查询/KG 语言", "spec"),
 "regex-chess":                 ("查询/KG 语言", "spec,reason"),
 "adaptive-rejection-sampler":  ("算法实现", "algo"),
 "distribution-search":         ("算法实现", "algo"),
 "largest-eigenval":            ("算法实现", "algo,perf"),
 "portfolio-optimization":      ("算法实现", "algo,perf"),
 "circuit-fibsqrt":             ("算法实现", "algo,hardware"),
 "schemelike-metacircular-eval":("算法实现", "algo,interpreter"),
 "llm-inference-batching-scheduler":("算法实现", "algo,perf"),
 "polyglot-c-py":               ("语言怪题/polyglot", "spec,adversarial"),
 "polyglot-rust-c":             ("语言怪题/polyglot", "spec,adversarial"),
 "gpt2-codegolf":               ("语言怪题/polyglot", "algo,perf"),
 "write-compressor":            ("语言怪题/polyglot", "algo,reveng"),
 "headless-terminal":           ("实现接口/规格", "codegen-spec"),
 "cancel-async-tasks":          ("实现接口/规格", "codegen-spec"),
 "constraints-scheduling":      ("实现接口/规格", "codegen-spec,reason"),
 "tune-mjcf":                   ("实现接口/规格", "domain,perf"),
 "torch-tensor-parallelism":    ("分布式/GPU 训练", "ml-torch"),
 "torch-pipeline-parallelism":  ("分布式/GPU 训练", "ml-torch"),
 "pytorch-model-cli":           ("分布式/GPU 训练", "ml-torch,spec"),
 "pytorch-model-recovery":      ("分布式/GPU 训练", "ml-torch,reveng"),
 "custom-memory-heap-crash":    ("C/C++ 调试", "debug-fix"),
 "fix-ocaml-gc":                ("C/C++ 调试", "debug-fix"),
 "qemu-startup":                ("虚拟化/QEMU", "sysadmin"),
 "qemu-alpine-ssh":             ("虚拟化/QEMU", "sysadmin"),
 "install-windows-3.11":        ("虚拟化/QEMU", "sysadmin"),
 "make-doom-for-mips":          ("模拟器/交叉编译", "reveng,perf"),
 "make-mips-interpreter":       ("模拟器/交叉编译", "reveng"),
 "feal-differential-cryptanalysis":("密码分析", "crypto"),
 "feal-linear-cryptanalysis":   ("密码分析", "crypto"),
 "overfull-hbox":               ("排版/LaTeX", "spec,tooling"),
 "large-scale-text-editing":    ("文本/编辑器", "spec,tooling"),
 "video-processing":            ("媒体处理", "vision,ffmpeg"),
 "path-tracing":                ("图形渲染", "algo"),
 "winning-avg-corewars":        ("对抗/博弈", "algo,search"),
}

by_env = collections.defaultdict(list)
for x in D:
    by_env[x["env"]].append(x)

miss = [e for e in by_env if e not in T]
if miss:
    print("UNLABELLED:", miss)

def cls(x):
    return x["cat"]

dom = collections.defaultdict(lambda: {"env": 0, "both2": 0, "split": 0, "zero": 0, "pass": 0, "n": 0})
skill = collections.defaultdict(lambda: {"env": 0, "pass": 0, "n": 0})
rows = []
for e, xs in sorted(by_env.items()):
    d, s = T.get(e, ("??", "??"))
    sc = sum(x["score"] for x in xs)
    n = len(xs)
    g = dom[d]; g["env"] += 1; g["pass"] += sc; g["n"] += n
    if sc == n: g["both2"] += 1
    elif sc == 0: g["zero"] += 1
    else: g["split"] += 1
    for sk in s.split(","):
        k = skill[sk]; k["env"] += 1; k["pass"] += sc; k["n"] += n
    rows.append((d, e, sc, n))

print("\n=== 领域聚合（pass@all = 通过样本/样本数）===")
print(f"{'domain':28s} {'env':>4s} {'2/2':>4s} {'1/2':>4s} {'0/2':>4s} {'pass@all':>9s}")
tot = [0, 0]
for d, g in sorted(dom.items(), key=lambda kv: -(kv[1]["pass"] / kv[1]["n"])):
    p = g["pass"]; n = g["n"]; tot[0] += p; tot[1] += n
    print(f"{d:28s} {g['env']:4d} {g['both2']:4d} {g['split']:4d} {g['zero']:4d} {p/n:8.3f}")
print(f"{'TOTAL':28s} {len(by_env):4d} {'':4s} {'':4s} {'':4s} {tot[0]/tot[1]:8.3f}")

print("\n=== 技能族聚合 ===")
for s, g in sorted(skill.items(), key=lambda kv: -(kv[1]["pass"] / kv[1]["n"])):
    print(f"{s:16s} env={g['env']:3d} pass@all={g['pass']/g['n']:.3f}")

print("\n=== 逐题（按分数）===")
for d, e, sc, n in sorted(rows, key=lambda r: (r[2], r[1])):
    print(f"{sc}/{n}  {d:24s} {e}")

# behaviour contrast
def agg(cats):
    xs = [x for x in D if x["cat"] in cats]
    out = {}
    for k in ["steps", "agent_min", "think_p50", "total_think", "compactions", "cmds",
              "repeat_ratio", "err_ratio", "explore_ratio", "write_cmds", "tests"]:
        v = [x[k] if k != "tests" else x["tests"][0] for x in xs]
        out[k] = statistics.median(v)
    out["n"] = len(xs)
    return out

print("\n=== 行为对比（中位数）===")
for name, cats in [("PASS", ["PASS"]), ("PARTIAL", ["DECLARED_PARTIAL"]),
                   ("ALLFAIL", ["DECLARED_ALLFAIL"]), ("TTL", ["TTL"]),
                   ("NONDONE", ["NONDONE", "NO_DECLARE", "CAP512"])]:
    a = agg(cats)
    print(f"{name:9s} n={a['n']:3d} steps={a['steps']:6.0f} min={a['agent_min']:6.0f} "
          f"think_p50={a['think_p50']:7.0f} tot_think={a['total_think']:9.0f} "
          f"compact={a['compactions']:4.0f} cmds={a['cmds']:5.0f} rep={a['repeat_ratio']:.2f} "
          f"err={a['err_ratio']:.2f} expl={a['explore_ratio']:.2f} write={a['write_cmds']:4.0f} test={a['tests']}")

# 只差一个测试的 partial
print("\n=== PARTIAL 中只差 1 个测试的条数 ===")
p1 = [x for x in D if x["cat"] == "DECLARED_PARTIAL" and x["tests"][1] == 1]
print(f"{len(p1)} / {sum(1 for x in D if x['cat']=='DECLARED_PARTIAL')}")
print("envs:", sorted({x["env"] for x in p1}))