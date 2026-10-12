# TAVIB: Tactile sensing time series data for robotic object interaction

TAVIB (Tactile Vibration) contains tactile time-series recordings from robotic object interaction experiments. The release is organized into two main data groups:

- `active_touch`: active vibro-tactile recordings from experimental objects, simulated objects, and strawberries.
- `slippage`: accelerometer and encoder recordings for texture, motion, and speed experiments with aluminum, plastic, and wood objects.

## Source data and related publications

TAVIB combines recordings from two previously reported experimental studies. The active-touch experimental, simulation, and strawberry data were previously deposited by Danissa Sandykbayeva and Valeriya Zhanuzakova in the [original OSF dataset](https://doi.org/10.17605/OSF.IO/G4AE8), associated with [Sandykbayeva et al., Scientific Reports (2026)](https://doi.org/10.1038/s41598-026-54766-9). The slippage and sliding recordings originate from the experiments described by [Massalim et al., Sensors (2020)](https://doi.org/10.3390/s20154121).

This combined release provides a unified directory structure, recording-level metadata, source-path mappings, and checksums. It does not represent a new data acquisition. Please cite the original datasets and publications as well as TAVIB when using these data.

## Repository structure

```text
TAVIB_dataset/
|-- README.md
|-- LICENSE
|-- data/
|   |-- active_touch/
|   |   |-- experimental/
|   |   |-- simulation/
|   |   |-- strawberries/
|   |-- slippage/
|   |   |-- aluminum/
|   |   |-- plastic/
|   |   |-- wood/
|-- metadata/
|   |-- file_index.csv
|   |-- path_mapping.csv
|-- checksums/
|   |-- sha256_manifest.csv
```

## Data contents

`data/active_touch/experimental/` contains 1750 CSV files organized by grasping force and object material:

```text
data/active_touch/experimental/<force>/<material>/cube<material>_<force>_trial<trial>.csv
```

Example:

```text
data/active_touch/experimental/3N/TPU/cubeTPU_3N_trial1.csv
```

`data/active_touch/simulation/` contains 1750 CSV files organized by simulated force and object material:

```text
data/active_touch/simulation/<force>/<material>/8000_<trial>.csv
```

Both manufactured-object subcollections contain 35 force/material conditions with 50 files per condition. The six elasticity materials were tested at 1-5 N; the additional PLA20, PLA40, PLA60, PLA80, and PLAhollow infill groups were tested only at 1 N. This is the intended acquisition coverage. The fully infilled PLA folder is named `PLA100` at every force in both subcollections.

`data/active_touch/strawberries/` contains 6000 CSV files organized by acquisition day and strawberry sample:

```text
data/active_touch/strawberries/<day>/<sample>/stm_sweep_<sweep>.csv
```

`data/slippage/` contains 36 CSV files organized by material, orientation, and speed:

```text
data/slippage/<material>/Orientation_<degrees>/speed_<speed_label>/data.csv
```

The slippage materials are `aluminum`, `plastic`, and `wood`. Orientations are `0`, `15`, `30`, and `45` degrees. Speed labels are `low`, `medium`, and `high`.

## File index and path mapping

`metadata/file_index.csv` contains one row per recording (9536 rows). It records the existing directory and filename labels; it does not add measurements or event annotations. All paths are relative to `TAVIB_dataset/` and use forward slashes. Empty fields indicate that a label does not apply to that subset.

| Column | Meaning |
| --- | --- |
| `path` | Current relative path of the recording |
| `subset` | `experimental`, `simulation`, `strawberries`, or `slippage` |
| `material` | Material folder label for manufactured objects or slippage recordings |
| `force_n` | Nominal force encoded by the manufactured-object force folder, in N |
| `day` | Strawberry acquisition-day number, 1-4 |
| `sample_id` | Strawberry folder identifier, `s1`-`s30` |
| `orientation_deg` | Slippage orientation encoded by the folder, in degrees |
| `motor_speed_label` | Slippage motor-speed folder label: `low`, `medium`, or `high` |
| `trial` | Trial number for manufactured objects or sweep number for strawberries |

`metadata/path_mapping.csv` also contains 9536 rows, with columns `original_path`, `new_path`, and `sha256`. It maps every recording's path before the organizational cleanup to its current path, including unchanged paths. The SHA-256 value is the same for the original and current file. Numbered simulation files are retained: the 50 files within each condition have identical contents, representing 35 distinct simulated responses overall.

The organizational cleanup renamed simulation `PLA` folders at 2-5 N to `PLA100` and removed the space before `.csv` in all simulation filenames. Operating-system metadata files are excluded from the release. Recording contents, headers, column order, and sample counts are unchanged.

## File formats and columns

All data files are comma-separated values files.

Experimental active-touch files use four columns:

| Column | Header in file | Meaning |
| --- | --- | --- |
| 1 | `Time (s)` | Receiver-side time stamp, in seconds |
| 2 | `AccelRight Value` | Receiver-side accelerometer signal as released |
| 3 | `Time (s)` | Transmitter-side time stamp, in seconds |
| 4 | `AccelLeft Value` | Transmitter-side accelerometer signal as released |

Simulated active-touch files use four columns:

| Column | Header in file | Meaning |
| --- | --- | --- |
| 1 | `Time1` | Transmitter-side time stamp, in seconds |
| 2 | `Transmitter` | Simulated transmitter signal |
| 3 | `Time2` | Receiver-side time stamp, in seconds |
| 4 | `Reciever` | Simulated receiver signal; the spelling is preserved from the released files |

Strawberry active-touch files use four columns:

| Column | Header in file | Meaning |
| --- | --- | --- |
| 1 | `sample` | Sequential sample index |
| 2 | `time_s` | Time stamp, in seconds |
| 3 | `acc_rx` | Receiver-side accelerometer ADC reading |
| 4 | `acc_tx` | Transmitter-side accelerometer ADC reading |

Slippage files are headerless CSV files with eight columns:

| Column | Suggested name | Meaning |
| --- | --- | --- |
| 1 | `accel_1_x` | First accelerometer x-axis, scaled to the released numeric range |
| 2 | `accel_1_y` | First accelerometer y-axis, scaled to the released numeric range |
| 3 | `accel_1_z` | First accelerometer z-axis, scaled to the released numeric range |
| 4 | `accel_2_x` | Second accelerometer x-axis, scaled to the released numeric range |
| 5 | `accel_2_y` | Second accelerometer y-axis, scaled to the released numeric range |
| 6 | `accel_2_z` | Second accelerometer z-axis, scaled to the released numeric range |
| 7 | `slip_velocity` | Slippage velocity computed from encoder readings, in m/s |
| 8 | `slide_velocity` | Sliding velocity computed from encoder readings, in m/s |

## Sampling

The active-touch and slippage recordings were acquired at 8 kHz.

## Checksums

The manifest contains 9540 entries covering all 9536 data CSV files, both metadata CSV files, `README.md`, and `LICENSE`. It excludes the manifest itself. Operating-system metadata files are not part of the release.

File integrity can be verified with:

```powershell
Import-Csv checksums/sha256_manifest.csv | ForEach-Object {
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.path).Hash.ToLowerInvariant()
  if ($actual -ne $_.sha256) { Write-Error "Checksum mismatch: $($_.path)" }
}
```

Run the command from the `TAVIB_dataset/` directory.

## License

The dataset is released under the Creative Commons Zero v1.0 Universal public domain dedication. See `LICENSE`.
