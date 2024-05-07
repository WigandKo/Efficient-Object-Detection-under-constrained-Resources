# Efficient-Object-Detection-under-constrained-Resources

| Network  | #Params | peak SRAM  | Epochs | mAP  | Latency |
| ----- | ----- |------- | ----- |------- | ----- |
|MbV2-r-w1.0-r224-D | 3.75M | 1.2MB | 145 | 44.24% | - |
|MbV2-r-w1.0-r224-T | 2.80M | 1.2MB | 170 | 46.53% | - |
|MbV2-r-w.35-r224-T | 0.84M | 0.487MB | 128 | 34.10% | - |
|MbV2-r-w0.7-r192-T | 1.62M | 0.507MB | 165 | 38.73% | 639ms |
|MbV2-r-w0.7-r224-T | 1.67M | 0.465MB | 241 | 45.16% | 783ms |
|MbV2-r-w0.7-r288-T | 1.66M | 0.474MB | 172 | 41.45% | 1077ms |
|MbV2-l-w0.7-r192-T | 1.67M | 0.499MB | 234 | 42.99% | 463ms |

| Network  | #Params | peak SRAM  | Epochs | mAP  | Latency |
| ----- | ----- |------- | ----- |------- | ----- |
|MCU-r-w1.0-r224-T |1.33M |0.483MB |156 |45.86% |1031ms|
|MCU-r-w1.0-r224-S |1.60M |0.483MB |167 |48.87% |1112ms|
|MCU-r-w1.0-r288-S |1.57M |0.499MB |159 |48.42% |1617ms|
|MCU-r-w1.0-r192-S |1.60M |0.494MB |184 |43.56% |864ms|
|MCU-l-w1.0-r192-S |1.59M |0.493MB |127 |41.02% |700ms|
