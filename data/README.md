# Data

The raw data is **not** included in this repository. It comes from the Kaggle competition
[Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)
(slug: `demand-forecasting-kernels-only`). Please respect the competition's terms of use.

## Manual download (no API needed)

1. Log in at kaggle.com and open the competition page.
2. Click **Join Competition** (or **Late Submission**) and accept the rules. The download button does not work until you do.
3. Open the **Data** tab and click **Download All** (about 5 MB zipped).
4. Unzip it and keep **`train.csv`** only. `test.csv` has no labels and is not used; this project builds its own hold-out from the last 90 days of `train.csv`.
5. Place the file at:

```
data/raw/train.csv
```

On Google Colab, upload `train.csv` to a Drive folder and point `PROJECT_DIR` in the notebook's config cell to it.

## Expected file

| Property | Value |
|---|---|
| Columns | `date`, `store`, `item`, `sales` |
| Rows | 913,000 |
| Date range | 2013-01-01 to 2017-12-31 (1,826 contiguous days) |
| Stores / items / series | 10 / 50 / 500 |
| Missing values / duplicate rows | 0 / 0 |
| Sales min / median / max | 0 / 47 / 231 |

If the notebook's first sanity report does not match this table, something is wrong with the file.
