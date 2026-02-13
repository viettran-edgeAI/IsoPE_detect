# Feature Selection Report

**Date**: 2026-02-13 13:53

## Summary

- Raw features loaded: **623**
- After variance filtering: **570** (removed 53)
- After correlation pruning: **533** (removed 37)
- After stability filtering: **475** (removed 58)
- Log-transformed features: **380**
- **Final selected: 475 features in 23 groups**

## Per-Group Summary

| Group | Raw | After Var | After Corr | After Stab |
|-------|-----|-----------|------------|------------|
| coff_header | 11 | 8 | 7 | 6 |
| data_directories | 32 | 28 | 22 | 20 |
| debug | 7 | 7 | 4 | 4 |
| delay_imports | 2 | 2 | 2 | 1 |
| dos_header | 4 | 1 | 1 | 1 |
| exports_hash | 32 | 32 | 32 | 32 |
| exports_scalar | 4 | 4 | 4 | 3 |
| file_metadata | 1 | 1 | 1 | 1 |
| imports_dll_hash | 64 | 58 | 58 | 22 |
| imports_func_hash | 256 | 256 | 256 | 255 |
| imports_scalar | 7 | 7 | 5 | 5 |
| load_config | 6 | 6 | 6 | 5 |
| optional_header | 39 | 33 | 29 | 24 |
| overlay | 4 | 4 | 4 | 4 |
| relocations | 3 | 3 | 2 | 2 |
| resources | 13 | 13 | 13 | 13 |
| rich_header | 6 | 6 | 4 | 4 |
| sections_aggregate | 7 | 7 | 4 | 4 |
| sections_counts | 4 | 4 | 3 | 2 |
| sections_name_hash | 32 | 16 | 16 | 12 |
| sections_per_slot | 80 | 66 | 55 | 50 |
| signatures | 4 | 4 | 2 | 2 |
| tls | 5 | 4 | 3 | 3 |

## Selected Features (475 total)

### coff_header (6 features)
- `coff_machine`
- `coff_num_sections`
- `coff_characteristics` [log1p]
- `coff_char_DLL`
- `coff_char_LARGE_ADDRESS_AWARE`
- `coff_char_DEBUG_STRIPPED`

### data_directories (20 features)
- `dd_EXPORT_TABLE_rva` [log1p]
- `dd_EXPORT_TABLE_size` [log1p]
- `dd_IMPORT_TABLE_rva` [log1p]
- `dd_IMPORT_TABLE_size` [log1p]
- `dd_EXCEPTION_TABLE_rva` [log1p]
- `dd_EXCEPTION_TABLE_size` [log1p]
- `dd_CERTIFICATE_TABLE_rva` [log1p]
- `dd_CERTIFICATE_TABLE_size`
- `dd_BASE_RELOCATION_TABLE_size` [log1p]
- `dd_DEBUG_DIR_rva` [log1p]
- `dd_DEBUG_DIR_size`
- `dd_TLS_TABLE_size` [log1p]
- `dd_LOAD_CONFIG_TABLE_rva` [log1p]
- `dd_BOUND_IMPORT_rva` [log1p]
- `dd_BOUND_IMPORT_size` [log1p]
- `dd_IAT_size` [log1p]
- `dd_DELAY_IMPORT_DESCRIPTOR_rva` [log1p]
- `dd_DELAY_IMPORT_DESCRIPTOR_size` [log1p]
- `dd_CLR_RUNTIME_HEADER_rva` [log1p]
- `dd_CLR_RUNTIME_HEADER_size`

### debug (4 features)
- `has_debug`
- `has_pogo`
- `has_repro`
- `has_pdb`

### delay_imports (1 features)
- `has_delay_imports`

### dos_header (1 features)
- `dos_e_lfanew`

### exports_hash (32 features)
- `exp_hash_0` [log1p]
- `exp_hash_1` [log1p]
- `exp_hash_2` [log1p]
- `exp_hash_3` [log1p]
- `exp_hash_4` [log1p]
- `exp_hash_5` [log1p]
- `exp_hash_6` [log1p]
- `exp_hash_7` [log1p]
- `exp_hash_8` [log1p]
- `exp_hash_9` [log1p]
- `exp_hash_10` [log1p]
- `exp_hash_11` [log1p]
- `exp_hash_12` [log1p]
- `exp_hash_13` [log1p]
- `exp_hash_14` [log1p]
- `exp_hash_15` [log1p]
- `exp_hash_16` [log1p]
- `exp_hash_17` [log1p]
- `exp_hash_18` [log1p]
- `exp_hash_19` [log1p]
- `exp_hash_20` [log1p]
- `exp_hash_21` [log1p]
- `exp_hash_22` [log1p]
- `exp_hash_23` [log1p]
- `exp_hash_24` [log1p]
- `exp_hash_25` [log1p]
- `exp_hash_26` [log1p]
- `exp_hash_27` [log1p]
- `exp_hash_28` [log1p]
- `exp_hash_29` [log1p]
- `exp_hash_30` [log1p]
- `exp_hash_31` [log1p]

### exports_scalar (3 features)
- `has_exports`
- `num_exports` [log1p]
- `export_name_count` [log1p]

### file_metadata (1 features)
- `file_size` [log1p]

### imports_dll_hash (22 features)
- `imp_dll_hash_4` [log1p]
- `imp_dll_hash_5` [log1p]
- `imp_dll_hash_6` [log1p]
- `imp_dll_hash_8` [log1p]
- `imp_dll_hash_9` [log1p]
- `imp_dll_hash_17` [log1p]
- `imp_dll_hash_24` [log1p]
- `imp_dll_hash_26` [log1p]
- `imp_dll_hash_27` [log1p]
- `imp_dll_hash_28` [log1p]
- `imp_dll_hash_34` [log1p]
- `imp_dll_hash_37` [log1p]
- `imp_dll_hash_39` [log1p]
- `imp_dll_hash_40` [log1p]
- `imp_dll_hash_43`
- `imp_dll_hash_48` [log1p]
- `imp_dll_hash_53`
- `imp_dll_hash_54` [log1p]
- `imp_dll_hash_57` [log1p]
- `imp_dll_hash_61` [log1p]
- `imp_dll_hash_62` [log1p]
- `imp_dll_hash_63` [log1p]

### imports_func_hash (255 features)
- `imp_func_hash_0` [log1p]
- `imp_func_hash_1` [log1p]
- `imp_func_hash_2` [log1p]
- `imp_func_hash_3` [log1p]
- `imp_func_hash_4` [log1p]
- `imp_func_hash_5` [log1p]
- `imp_func_hash_6` [log1p]
- `imp_func_hash_7` [log1p]
- `imp_func_hash_8` [log1p]
- `imp_func_hash_9` [log1p]
- `imp_func_hash_10` [log1p]
- `imp_func_hash_11` [log1p]
- `imp_func_hash_12` [log1p]
- `imp_func_hash_13` [log1p]
- `imp_func_hash_14` [log1p]
- `imp_func_hash_15`
- `imp_func_hash_16` [log1p]
- `imp_func_hash_17` [log1p]
- `imp_func_hash_18` [log1p]
- `imp_func_hash_19`
- `imp_func_hash_20` [log1p]
- `imp_func_hash_21` [log1p]
- `imp_func_hash_22` [log1p]
- `imp_func_hash_23` [log1p]
- `imp_func_hash_24`
- `imp_func_hash_25` [log1p]
- `imp_func_hash_26` [log1p]
- `imp_func_hash_27` [log1p]
- `imp_func_hash_28` [log1p]
- `imp_func_hash_29`
- `imp_func_hash_30` [log1p]
- `imp_func_hash_31` [log1p]
- `imp_func_hash_32` [log1p]
- `imp_func_hash_33` [log1p]
- `imp_func_hash_34`
- `imp_func_hash_35` [log1p]
- `imp_func_hash_36` [log1p]
- `imp_func_hash_37` [log1p]
- `imp_func_hash_38` [log1p]
- `imp_func_hash_39` [log1p]
- `imp_func_hash_40` [log1p]
- `imp_func_hash_41` [log1p]
- `imp_func_hash_42` [log1p]
- `imp_func_hash_43` [log1p]
- `imp_func_hash_44` [log1p]
- `imp_func_hash_45` [log1p]
- `imp_func_hash_46` [log1p]
- `imp_func_hash_47` [log1p]
- `imp_func_hash_48` [log1p]
- `imp_func_hash_49` [log1p]
- `imp_func_hash_50` [log1p]
- `imp_func_hash_51` [log1p]
- `imp_func_hash_52` [log1p]
- `imp_func_hash_53` [log1p]
- `imp_func_hash_54` [log1p]
- `imp_func_hash_55` [log1p]
- `imp_func_hash_56` [log1p]
- `imp_func_hash_57` [log1p]
- `imp_func_hash_58` [log1p]
- `imp_func_hash_59` [log1p]
- `imp_func_hash_60` [log1p]
- `imp_func_hash_61` [log1p]
- `imp_func_hash_62`
- `imp_func_hash_63` [log1p]
- `imp_func_hash_64` [log1p]
- `imp_func_hash_65`
- `imp_func_hash_66` [log1p]
- `imp_func_hash_67` [log1p]
- `imp_func_hash_68` [log1p]
- `imp_func_hash_69` [log1p]
- `imp_func_hash_70` [log1p]
- `imp_func_hash_71` [log1p]
- `imp_func_hash_72` [log1p]
- `imp_func_hash_73` [log1p]
- `imp_func_hash_74` [log1p]
- `imp_func_hash_75` [log1p]
- `imp_func_hash_76` [log1p]
- `imp_func_hash_77` [log1p]
- `imp_func_hash_78` [log1p]
- `imp_func_hash_79` [log1p]
- `imp_func_hash_80` [log1p]
- `imp_func_hash_81` [log1p]
- `imp_func_hash_82` [log1p]
- `imp_func_hash_83` [log1p]
- `imp_func_hash_84` [log1p]
- `imp_func_hash_85` [log1p]
- `imp_func_hash_86` [log1p]
- `imp_func_hash_87` [log1p]
- `imp_func_hash_88` [log1p]
- `imp_func_hash_89` [log1p]
- `imp_func_hash_90` [log1p]
- `imp_func_hash_91` [log1p]
- `imp_func_hash_92` [log1p]
- `imp_func_hash_93` [log1p]
- `imp_func_hash_94` [log1p]
- `imp_func_hash_95` [log1p]
- `imp_func_hash_96` [log1p]
- `imp_func_hash_97`
- `imp_func_hash_98` [log1p]
- `imp_func_hash_99` [log1p]
- `imp_func_hash_100` [log1p]
- `imp_func_hash_101` [log1p]
- `imp_func_hash_102` [log1p]
- `imp_func_hash_103` [log1p]
- `imp_func_hash_104` [log1p]
- `imp_func_hash_105` [log1p]
- `imp_func_hash_106` [log1p]
- `imp_func_hash_107` [log1p]
- `imp_func_hash_108` [log1p]
- `imp_func_hash_109` [log1p]
- `imp_func_hash_110` [log1p]
- `imp_func_hash_111` [log1p]
- `imp_func_hash_112` [log1p]
- `imp_func_hash_113`
- `imp_func_hash_114`
- `imp_func_hash_115` [log1p]
- `imp_func_hash_116` [log1p]
- `imp_func_hash_117` [log1p]
- `imp_func_hash_118` [log1p]
- `imp_func_hash_119` [log1p]
- `imp_func_hash_120` [log1p]
- `imp_func_hash_121` [log1p]
- `imp_func_hash_122` [log1p]
- `imp_func_hash_123` [log1p]
- `imp_func_hash_124` [log1p]
- `imp_func_hash_125` [log1p]
- `imp_func_hash_126` [log1p]
- `imp_func_hash_127` [log1p]
- `imp_func_hash_128` [log1p]
- `imp_func_hash_129` [log1p]
- `imp_func_hash_130` [log1p]
- `imp_func_hash_131` [log1p]
- `imp_func_hash_132` [log1p]
- `imp_func_hash_133` [log1p]
- `imp_func_hash_134` [log1p]
- `imp_func_hash_135`
- `imp_func_hash_136` [log1p]
- `imp_func_hash_137` [log1p]
- `imp_func_hash_138` [log1p]
- `imp_func_hash_139` [log1p]
- `imp_func_hash_140` [log1p]
- `imp_func_hash_141`
- `imp_func_hash_142` [log1p]
- `imp_func_hash_143` [log1p]
- `imp_func_hash_144` [log1p]
- `imp_func_hash_145` [log1p]
- `imp_func_hash_146` [log1p]
- `imp_func_hash_147` [log1p]
- `imp_func_hash_148` [log1p]
- `imp_func_hash_149` [log1p]
- `imp_func_hash_150` [log1p]
- `imp_func_hash_151` [log1p]
- `imp_func_hash_152` [log1p]
- `imp_func_hash_153` [log1p]
- `imp_func_hash_154` [log1p]
- `imp_func_hash_155` [log1p]
- `imp_func_hash_156` [log1p]
- `imp_func_hash_157` [log1p]
- `imp_func_hash_158` [log1p]
- `imp_func_hash_159` [log1p]
- `imp_func_hash_160` [log1p]
- `imp_func_hash_161` [log1p]
- `imp_func_hash_162` [log1p]
- `imp_func_hash_163` [log1p]
- `imp_func_hash_164` [log1p]
- `imp_func_hash_165` [log1p]
- `imp_func_hash_166` [log1p]
- `imp_func_hash_168` [log1p]
- `imp_func_hash_169` [log1p]
- `imp_func_hash_170`
- `imp_func_hash_171` [log1p]
- `imp_func_hash_172` [log1p]
- `imp_func_hash_173` [log1p]
- `imp_func_hash_174` [log1p]
- `imp_func_hash_175` [log1p]
- `imp_func_hash_176` [log1p]
- `imp_func_hash_177`
- `imp_func_hash_178`
- `imp_func_hash_179` [log1p]
- `imp_func_hash_180` [log1p]
- `imp_func_hash_181` [log1p]
- `imp_func_hash_182` [log1p]
- `imp_func_hash_183` [log1p]
- `imp_func_hash_184` [log1p]
- `imp_func_hash_185` [log1p]
- `imp_func_hash_186` [log1p]
- `imp_func_hash_187` [log1p]
- `imp_func_hash_188` [log1p]
- `imp_func_hash_189` [log1p]
- `imp_func_hash_190` [log1p]
- `imp_func_hash_191` [log1p]
- `imp_func_hash_192` [log1p]
- `imp_func_hash_193` [log1p]
- `imp_func_hash_194` [log1p]
- `imp_func_hash_195` [log1p]
- `imp_func_hash_196` [log1p]
- `imp_func_hash_197` [log1p]
- `imp_func_hash_198` [log1p]
- `imp_func_hash_199` [log1p]
- `imp_func_hash_200` [log1p]
- `imp_func_hash_201`
- `imp_func_hash_202` [log1p]
- `imp_func_hash_203` [log1p]
- `imp_func_hash_204` [log1p]
- `imp_func_hash_205` [log1p]
- `imp_func_hash_206` [log1p]
- `imp_func_hash_207` [log1p]
- `imp_func_hash_208` [log1p]
- `imp_func_hash_209` [log1p]
- `imp_func_hash_210` [log1p]
- `imp_func_hash_211` [log1p]
- `imp_func_hash_212` [log1p]
- `imp_func_hash_213` [log1p]
- `imp_func_hash_214` [log1p]
- `imp_func_hash_215`
- `imp_func_hash_216` [log1p]
- `imp_func_hash_217` [log1p]
- `imp_func_hash_218` [log1p]
- `imp_func_hash_219` [log1p]
- `imp_func_hash_220` [log1p]
- `imp_func_hash_221` [log1p]
- `imp_func_hash_222` [log1p]
- `imp_func_hash_223` [log1p]
- `imp_func_hash_224` [log1p]
- `imp_func_hash_225` [log1p]
- `imp_func_hash_226` [log1p]
- `imp_func_hash_227` [log1p]
- `imp_func_hash_228`
- `imp_func_hash_229` [log1p]
- `imp_func_hash_230` [log1p]
- `imp_func_hash_231` [log1p]
- `imp_func_hash_232` [log1p]
- `imp_func_hash_233` [log1p]
- `imp_func_hash_234` [log1p]
- `imp_func_hash_235` [log1p]
- `imp_func_hash_236` [log1p]
- `imp_func_hash_237` [log1p]
- `imp_func_hash_238` [log1p]
- `imp_func_hash_239` [log1p]
- `imp_func_hash_240` [log1p]
- `imp_func_hash_241` [log1p]
- `imp_func_hash_242` [log1p]
- `imp_func_hash_243` [log1p]
- `imp_func_hash_244` [log1p]
- `imp_func_hash_245` [log1p]
- `imp_func_hash_246` [log1p]
- `imp_func_hash_247` [log1p]
- `imp_func_hash_248` [log1p]
- `imp_func_hash_249` [log1p]
- `imp_func_hash_250` [log1p]
- `imp_func_hash_251` [log1p]
- `imp_func_hash_252` [log1p]
- `imp_func_hash_253` [log1p]
- `imp_func_hash_254` [log1p]
- `imp_func_hash_255` [log1p]

### imports_scalar (5 features)
- `has_imports`
- `num_import_dlls`
- `num_import_by_ordinal` [log1p]
- `imphash` [log1p]
- `num_suspicious_imports`

### load_config (5 features)
- `has_load_config`
- `lc_size`
- `lc_security_cookie` [log1p]
- `lc_seh_count` [log1p]
- `lc_guard_flags` [log1p]

### optional_header (24 features)
- `opt_major_linker`
- `opt_minor_linker`
- `opt_sizeof_init_data` [log1p]
- `opt_sizeof_uninit_data` [log1p]
- `opt_entrypoint` [log1p]
- `opt_imagebase`
- `opt_section_alignment`
- `opt_file_alignment` [log1p]
- `opt_major_os_ver`
- `opt_minor_os_ver` [log1p]
- `opt_major_subsys_ver`
- `opt_minor_subsys_ver`
- `opt_sizeof_image` [log1p]
- `opt_sizeof_headers` [log1p]
- `opt_checksum` [log1p]
- `opt_subsystem` [log1p]
- `opt_dll_characteristics`
- `opt_sizeof_heap_commit`
- `opt_dllchar_DYNAMIC_BASE`
- `opt_dllchar_NX_COMPAT`
- `opt_dllchar_GUARD_CF`
- `opt_dllchar_HIGH_ENTROPY_VA`
- `opt_dllchar_NO_SEH`
- `checksum_matches`

### overlay (4 features)
- `has_overlay`
- `overlay_size` [log1p]
- `overlay_ratio`
- `overlay_entropy`

### relocations (2 features)
- `has_relocations`
- `num_relocation_blocks` [log1p]

### resources (13 features)
- `has_resources`
- `rsrc_num_directories` [log1p]
- `rsrc_num_data_entries` [log1p]
- `rsrc_total_size` [log1p]
- `rsrc_size_ratio` [log1p]
- `rsrc_num_types` [log1p]
- `rsrc_has_manifest`
- `rsrc_has_version`
- `rsrc_has_icons`
- `rsrc_has_dialogs`
- `rsrc_has_string_table`
- `rsrc_mean_entropy` [log1p]
- `rsrc_max_entropy` [log1p]

### rich_header (4 features)
- `rich_key` [log1p]
- `rich_mean_id`
- `rich_max_build_id`
- `rich_total_count` [log1p]

### sections_aggregate (4 features)
- `sec_mean_entropy`
- `sec_max_entropy`
- `sec_min_entropy`
- `sec_std_entropy`

### sections_counts (2 features)
- `num_exec_sections` [log1p]
- `num_write_sections`

### sections_name_hash (12 features)
- `sec_name_hash_2` [log1p]
- `sec_name_hash_3` [log1p]
- `sec_name_hash_8`
- `sec_name_hash_15` [log1p]
- `sec_name_hash_16` [log1p]
- `sec_name_hash_17`
- `sec_name_hash_20` [log1p]
- `sec_name_hash_23`
- `sec_name_hash_25` [log1p]
- `sec_name_hash_26` [log1p]
- `sec_name_hash_30` [log1p]
- `sec_name_hash_31` [log1p]

### sections_per_slot (50 features)
- `sec0_entropy`
- `sec0_vsize` [log1p]
- `sec0_vr_ratio`
- `sec0_is_exec`
- `sec0_is_write`
- `sec0_has_code`
- `sec1_entropy`
- `sec1_vsize` [log1p]
- `sec1_is_exec`
- `sec1_has_code`
- `sec2_entropy`
- `sec2_vsize` [log1p]
- `sec2_rsize` [log1p]
- `sec2_characteristics` [log1p]
- `sec2_is_exec`
- `sec3_entropy`
- `sec3_vsize` [log1p]
- `sec3_characteristics` [log1p]
- `sec3_is_write`
- `sec4_entropy`
- `sec4_vsize` [log1p]
- `sec4_vr_ratio` [log1p]
- `sec4_characteristics` [log1p]
- `sec4_is_write`
- `sec5_entropy` [log1p]
- `sec5_vsize` [log1p]
- `sec5_rsize` [log1p]
- `sec5_vr_ratio` [log1p]
- `sec5_characteristics` [log1p]
- `sec5_is_write`
- `sec6_entropy` [log1p]
- `sec6_vsize` [log1p]
- `sec6_rsize` [log1p]
- `sec6_vr_ratio` [log1p]
- `sec6_characteristics` [log1p]
- `sec6_is_write`
- `sec7_entropy` [log1p]
- `sec7_vsize` [log1p]
- `sec7_vr_ratio` [log1p]
- `sec7_characteristics` [log1p]
- `sec7_is_write`
- `sec8_entropy` [log1p]
- `sec8_rsize` [log1p]
- `sec8_vr_ratio` [log1p]
- `sec8_characteristics` [log1p]
- `sec9_entropy` [log1p]
- `sec9_rsize` [log1p]
- `sec9_vr_ratio` [log1p]
- `sec9_characteristics` [log1p]
- `sec9_is_write`

### signatures (2 features)
- `num_certificates`
- `sig_verified`

### tls (3 features)
- `tls_num_callbacks` [log1p]
- `tls_characteristics` [log1p]
- `tls_data_size` [log1p]

## Malware Validation/Test Separation

- Validation size before control: **2998**
- Validation size after control: **400**
- Exact duplicates removed: **0**
- Near-duplicates removed (hash_vector_cosine): **2598**
- Shared imphash removed: **1488**
- Downsample removed: **2598**
- Target range (5-8% benign test): **[250, 400]**
- Actual validation ratio to benign test: **0.0801**