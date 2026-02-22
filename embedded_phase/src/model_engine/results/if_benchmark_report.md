# Embedded Inference Benchmark Report

Benchmark scope: 10 files from `BENIGN_TEST_DATASET` and 10 files from `MALWARE_TEST_DATASET`.

Measured stages:
- File size
- Feature extraction time from `lief_feature_extractor` (`processing_time_ms`)
- Model inference time: standardization + quantization + IF scoring
- RAM usage (current process RSS in MB)

| Split | File | File Size (KB) | Feature Extraction Time (ms) | Inference Time (ms) | RAM Usage (MB RSS) | Score | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| benign_test | 000d81f6459bd4652a259aa98b6bc82bd9053045a1199f340727868bd887ec00.dll | 2956.031 | 39.926 | 0.144 | 18.656 | -0.124 | anomaly |
| benign_test | 000de8f56588e1a54bbd2d07cd2dff3e9967fcc42bc55a9fb476fedf66e4d9c2.dll | 60.305 | 2.909 | 0.148 | 18.781 | -0.097 | anomaly |
| benign_test | 000e0a55cd58e57306f65621bf03dddd4b721715bf4be4f45239c01043138243.dll | 84.305 | 2.754 | 0.151 | 18.781 | -0.106 | anomaly |
| benign_test | 000f278b7eed4ce262aefff1fd280b549999837ec9f89f44b3ef970162c8431e.dll | 53.969 | 2.735 | 0.155 | 18.781 | -0.100 | anomaly |
| benign_test | 001672675ff4a5dc32b64ecd126acd599fd20d53ab156251e883f6f51aa91fb2.dll | 8.500 | 0.547 | 0.138 | 18.781 | -0.114 | anomaly |
| benign_test | 001d74d5028821dc76f07184ed3fb37df6c774d360b68ef13ba619d5ab55f521.dll | 226.500 | 2.172 | 0.160 | 18.781 | -0.079 | anomaly |
| benign_test | 001dbc4c67d6b135dc49768191d8c78bb90b317c466355fce2e57f20c54ad75c.dll | 75.603 | 1.423 | 0.177 | 18.781 | -0.108 | anomaly |
| benign_test | 002a2f5cfcf551f27ef4637480c60fc7ceeb53ade3233f8645d89ffdb39a3a58.exe | 21.000 | 0.745 | 0.215 | 18.781 | -0.123 | anomaly |
| benign_test | 0034d97ee6145439793894b3ac7048d310ccdbc9bbd0c9c257de56363c018089.exe | 33.828 | 2.814 | 0.195 | 18.781 | -0.092 | anomaly |
| benign_test | 003df7586997a20ff38bb2b922a340f0a8c54e4a1c130b3c730a81f32dade0fe.dll | 19.070 | 1.614 | 0.213 | 18.781 | -0.108 | anomaly |
| malware_test | 0005626a93719fa8620cf1ed3be816cbb80f11d4b8e99985f25bb7e3175af6a0.exe | 116.000 | 1.674 | 0.192 | 18.781 | -0.132 | anomaly |
| malware_test | 0009f3b69873758c4a6472f4169c8cdc9b44a76e666df1b5f5de66ed91c552dc.exe | 5258.000 | 43.971 | 0.116 | 18.781 | -0.136 | anomaly |
| malware_test | 0019e0aeb5f6afec763e9e8c237b6f64cddfb4e8d7cba98cfdc0ba0d569d2460.exe | 5266.500 | 42.526 | 0.114 | 18.781 | -0.136 | anomaly |
| malware_test | 0049bd68937a94dc047a1ad06222fceb30315d6d26546a9a2665453045bd8403.exe | 3194.000 | 17.572 | 0.118 | 18.781 | -0.125 | anomaly |
| malware_test | 00654e2183b5d32d6676fe6971c82eb511e4ae785c352e2fc03af5ec30f72e6c.exe | 4426.500 | 31.674 | 0.135 | 18.781 | -0.117 | anomaly |
| malware_test | 006622b9cb14dd2dc52f7f52e800f6a4da24330f4102810b86c414f843846752.exe | 10240.000 | 58.208 | 0.151 | 18.781 | -0.108 | anomaly |
| malware_test | 00757772d873cfdf11241fc390c5ad2d2b3a27648656fe6ca502b12d9598cd6e.exe | 7070.400 | 52.863 | 0.151 | 18.781 | -0.113 | anomaly |
| malware_test | 008902cb35391208a2c1ceecf2c10afdf17f938c9b4cae1a92843e9875cc1a1c.exe | 8.000 | 0.542 | 0.118 | 18.781 | -0.125 | anomaly |
| malware_test | 00a16089397d26dec07ba75d5ac027fba40482e0af441942d8de1475aa133aab.exe | 45289.000 | 241.279 | 0.164 | 18.781 | -0.095 | anomaly |
| malware_test | 00a22dc8a9c7fc5a72ce60eabae3304607e45a7212dcf20cad9a0e657fde8092.exe | 10025.500 | 49.368 | 0.127 | 18.781 | -0.125 | anomaly |

- Average feature extraction time: `29.866 ms`
- Average inference time (standardization + quantization + inference): `0.154 ms`
- Peak observed RSS during benchmark loop: `18.781 MB`
