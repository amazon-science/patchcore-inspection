"""実験スクリプト共通：タイムスタンプ付き出力ディレクトリと再現メタ情報。

CLAUDE.md「実行・結果出力の規約」に従い、各実行を
`<base>/<YYYYMMDD_HHMMSS>/`（JST）に隔離して既存結果を上書きせず、
再現用の `meta.json` と実行ログ `log.txt` を残すための小ユーティリティ。
"""
import atexit
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))


def jst_timestamp() -> str:
    """JST の `YYYYMMDD_HHMMSS` 文字列。出力フォルダ名の規約書式。"""
    return datetime.now(JST).strftime("%Y%m%d_%H%M%S")


def make_run_dir(base: str, timestamp: str = None) -> str:
    """`base/<timestamp>/` を作成してパスを返す（既存を上書きしない）。"""
    ts = timestamp or jst_timestamp()
    run_dir = os.path.join(base, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def git_commit_hash() -> str:
    """現在の git commit hash（取得失敗時は None）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _versions() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        import faiss

        info["faiss"] = getattr(faiss, "__version__", "unknown")
    except Exception:
        pass
    return info


def write_meta(run_dir: str, extra: dict = None) -> dict:
    """再現メタ情報を `run_dir/meta.json` に書き出す。

    実行コマンド・git commit・JST 時刻・主要ライブラリ版を記録し、
    `extra` に渡したハイパーパラメータ等をマージする。
    """
    meta = {
        "command": " ".join(sys.argv),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "git_commit": git_commit_hash(),
        "timestamp_jst": jst_timestamp(),
        "versions": _versions(),
    }
    if extra:
        meta["params"] = extra
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


class _Fork:
    """write を2つのストリームへ複製する薄いラッパ。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


class Tee:
    """stdout/stderr を `run_dir/log.txt` にも複製するコンテキストマネージャ。"""

    def __init__(self, run_dir: str, filename: str = "log.txt"):
        self.log_path = os.path.join(run_dir, filename)
        self._file = None
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        self._file = open(self.log_path, "w")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout = _Fork(self._stdout, self._file)
        sys.stderr = _Fork(self._stderr, self._file)
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._file.close()
        return False


def start_tee(run_dir: str, filename: str = "log.txt"):
    """sys.stdout/stderr を `run_dir/log.txt` にも複製し、ファイルを返す。

    本体を `with` で囲みにくい click ベースのスクリプト向け。プロセス終了時に
    atexit で元の stdout/stderr を復元する（ファイルは明示クローズしない＝
    logging のシャットダウン flush と競合させないため）。返り値のファイルを
    `logging.StreamHandler` に渡せば LOGGER 出力も同じ log.txt に集約できる。
    """
    f = open(os.path.join(run_dir, filename), "a")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Fork(orig_out, f)
    sys.stderr = _Fork(orig_err, f)

    def _restore():
        sys.stdout, sys.stderr = orig_out, orig_err
        try:
            f.flush()
        except Exception:
            pass

    atexit.register(_restore)
    return f
