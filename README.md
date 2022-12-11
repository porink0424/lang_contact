# Language Contact

## Run

### Training

After setting the parameters to `train_batcher.sh`, please run:

```
bash train_batcher.sh
```

Then you will get result files in `result` directory.

### Organizing Data

You can make raw result files to visualized results (multiple results at one time) by running:

```
bash organize_data.sh $natt $nval $cvoc $clen id1 id2 ...
```

Then you will get the visualized results as a markdown file in `result_md` directory.

### [Optinal] topsim

When the input size is big, it may be hard to calculate topsim in GPU environment. If you want to run in the CPU environment, run:

```
bash topsim.sh id natt nval cvoc clen
```

### Get averaged results

After making a markdown file, you can calculate averaged results by running:

```
bash average.sh id1 id2 id3 ...
```

## commit message prefix

- feat: 新しい機能
- fix: バグの修正
- docs: ドキュメントのみの変更
- style: 空白、フォーマット、セミコロン追加など
- refactor: 仕様に影響がないコード改善(リファクタ)
- perf: パフォーマンス向上関連
- test: テスト関連
- chore: ビルド、補助ツール、ライブラリ関連
