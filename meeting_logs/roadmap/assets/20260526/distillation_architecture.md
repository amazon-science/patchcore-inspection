# 蒸留アーキテクチャ（教師バンク → 生徒モデル）

教師 PatchCore のメモリバンクが定義する距離写像 `x → 最近傍 L2 距離` を、
定数サイズの生徒モデル（M3 系 MLP / M4 プロトタイプ）に回帰学習で写し取る。

- **生徒の入力 `x`**：教師バンク `bank[idx]` ＋ガウス摂動で生成した合成パッチ特徴 `[D]`
- **生徒の出力ターゲット `d`**：教師バンク `IndexFlatL2` への最近傍 squared-L2 距離（＝教師の異常スコア）
- **更新対象**：生徒の重みのみ（教師バンクは凍結・参照のみ）

```mermaid
flowchart TB
    subgraph TEACHER["🎓 教師 PatchCore（凍結）"]
        BANK["メモリバンク<br/>faiss IndexFlatL2<br/><b>bank: [N × D]</b><br/>(正常パッチ特徴の集合)"]
        STAT["バンク統計<br/>feat_mean, feat_std"]
        BANK -->|"reconstruct_n"| STAT
    end

    subgraph GEN["合成クエリ生成 (データフリー)"]
        Q["<b>x = bank[idx] + N(0, σ·std)</b><br/>(正常近傍)<br/>＋ 遠方点 far (異常側 10%)"]
    end

    subgraph LABEL["教師による正解付け = 教師信号"]
        NN["bank への最近傍探索<br/>label_index.search(x, 1)<br/><b>d = min‖x − bank‖² (squared L2)</b>"]
    end

    subgraph STUDENT["🧑‍🎓 生徒モデル (定数サイズ)"]
        MLP["StudentMLP / PrototypeNet<br/>pred = f(x)"]
    end

    BANK -->|"① 入力 x の種"| Q
    STAT --> Q
    BANK  -->|"② 正解 d を引く参照テーブル"| NN
    Q -->|"<b>生徒の入力 x [D]</b>"| MLP
    Q -->|"同じ x を教師に渡す"| NN
    NN -->|"<b>生徒の出力ターゲット d</b><br/>target = log(d+eps)"| LOSS

    MLP -->|"pred"| LOSS["損失<br/>huber: ‖pred − log d‖<br/>ranking: 順位一致"]
    LOSS -.->|"逆伝播で更新するのは<br/><b>生徒の重みのみ</b>(教師は凍結)"| MLP

    classDef teacher fill:#e8f0ff,stroke:#3b6fb0,color:#000;
    classDef student fill:#fff0e8,stroke:#c0703b,color:#000;
    classDef io fill:#eafbea,stroke:#3ba05a,color:#000;
    class TEACHER,BANK,STAT teacher;
    class STUDENT,MLP student;
    class LABEL,NN io;
```

## 入出力の対応表

| 役割 | 実体 | 教師のどこ由来か |
|------|------|------------------|
| 生徒の **入力 `x`** | パッチ特徴ベクトル `[D]` | 教師バンク `bank[idx]` ＋ガウス摂動（`train_student.py` L71-82） |
| 生徒の **出力ターゲット `d`** | 最近傍 squared-L2 距離（＝教師の異常スコア） | 教師バンク `IndexFlatL2` への `search`（`train_student.py` L84） |
| 学習で更新されるもの | **生徒の重みだけ** | 教師バンクは凍結・参照のみ |

核心：生徒が写し取るのは教師ネットの重みではなく、「教師メモリバンクが定義する距離写像
`x → 最近傍距離`」。バンク `[N×D]` が①入力 `x` の生成源と②正解 `d` を引く参照テーブルの
両方を兼ねている。
</content>
</invoke>
