### Qwen3-VL-4B
```
"metrics": {
    "success": 93,
    "total": 100,
    "failed_json": 1,
    "total_reward": 93.0
  }
```
```
"metrics": {
    "success": 94,
    "total": 100,
    "failed_json": 0,
    "total_reward": 94.0
  }
```
```(改prompt且没有改数据的)
"metrics": {
    "success": 88,
    "total": 100,
    "failed_json": 5,
    "total_reward": 88.0
  }
```
(holo2_screenspot_v2_20260122_222931)
``` (89.44)
"metrics": {
    "success": 161,
    "total": 180,
    "failed_json": 10,
    "total_reward": 161.0
  }
```

### Qwen3-VL-8B

```
"metrics": {
    "accuracy": 0.91,
    "avg_steps": 1.01,
    "avg_reward": 0.8483643387665909,
    "failed_breakdown": {
      "1": 3,
      "2": 0,
      "3": 4,
      "4": 2
    }
  }
```
```
"metrics": {
    "accuracy": 0.9,
    "avg_steps": 1.01,
    "avg_reward": 0.8399429565734878,
    "failed_breakdown": {
      "1": 2,
      "2": 0,
      "3": 6,
      "4": 2
    }
  }
```
(grounding_baseline_20260122_212810_relax7)
```
"metrics": {
    "accuracy": 0.9333333333333333,
    "avg_steps": 1.0166666666666666,
    "avg_reward": 0.8902338910044046,
    "failed_breakdown": {
      "1": 3,
      "2": 0,
      "3": 8,
      "4": 1
    }
  }
```

### Qwen3-VL-235B-A22B

```
"metrics": {
    "accuracy": 0.93,
    "avg_steps": 1.0,
    "avg_reward": 0.8900678142056156,
    "failed_breakdown": {
      "1": 1,
      "2": 0,
      "3": 5,
      "4": 1
    }
  }
```
```
"metrics": {
    "accuracy": 0.92,
    "avg_steps": 1.0,
    "avg_reward": 0.8751155503914818,
    "failed_breakdown": {
      "1": 1,
      "2": 0,
      "3": 6,
      "4": 1
    }
  }
```
(grounding_baseline_20260122_213719_relax7)
```
"metrics": {
    "accuracy": 0.9166666666666666,
    "avg_steps": 1.0,
    "avg_reward": 0.8558789225460076,
    "failed_breakdown": {
      "1": 7,
      "2": 0,
      "3": 7,
      "4": 1
    }
  }
```

### Qwen3-VL-235B-A22B-PathFinding

```
"metrics": {
    "accuracy": 0.38,
    "avg_steps": 7.31,
    "avg_reward": 0.047150368899366725,
    "failed_breakdown": {
      "1": 0,
      "2": 4,
      "3": 47,
      "4": 11
    }
  },
```

### SFT2 (ScreenSpot)
```
"metrics": {
    "accuracy": 0.8,
    "avg_steps": 9.95,
    "avg_reward": 0.7028194488110627,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 19,
      "4": 1
    }
  }
```
(trained_20260122_212539_relax7)

### SFT3 (ScreenSpot-Pro)
```
"metrics": {
    "accuracy": 0.87,
    "avg_steps": 10.05,
    "avg_reward": 0.8052725043120172,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 12,
      "4": 1
    }
  }
```
(trained_20260122_212458_relax7)

<!-- ### SFT4 (Mind2Web)
```
"metrics": {
    "accuracy": 0.6,
    "avg_steps": 7.01,
    "avg_reward": 0.3780764351739002,
    "failed_breakdown": {
      "1": 6,
      "2": 0,
      "3": 34,
      "4": 0
    }
  }
``` -->

### 结果文件夹

holo2_screenspot_v2_20260122_160057（4b）
grounding_baseline_20260121_231921_relax7（8b）
grounding_baseline_20260121_232534_relax7（235b）
api_baseline_20260121_233102_relax7（235b_pathfinding）
trained_20260121_230736_relax7（sft2）
trained_20260121_234511_relax7（sft3）
trained_20260121_234850_relax7（sft4）

### Ablation-1200

#### Qwen3-VL-4B (holo2_screenspot_20260122_171014)
```
"metrics": {
    "success": 90,
    "total": 100,
    "failed_json": 2,
    "total_reward": 90.0
  }
```

#### Qwen3-VL-8B (grounding_baseline_20260122_165721_relax7)
```
"metrics": {
    "accuracy": 0.88,
    "avg_steps": 1.01,
    "avg_reward": 0.8031560610657994,
    "failed_breakdown": {
      "1": 4,
      "2": 0,
      "3": 7,
      "4": 1
    }
  }
```

#### Qwen3-VL-235B-A22B (grounding_baseline_20260122_170133_relax7)
```
"metrics": {
    "accuracy": 0.91,
    "avg_steps": 1.0,
    "avg_reward": 0.8513638606129732,
    "failed_breakdown": {
      "1": 3,
      "2": 0,
      "3": 6,
      "4": 0
    }
  }
```

#### Qwen3-VL-235B-A22B-PathFinding (api_baseline_20260122_171904_relax7)
```
"metrics": {
    "accuracy": 0.38,
    "avg_steps": 6.69,
    "avg_reward": 0.06012634534071262,
    "failed_breakdown": {
      "1": 0,
      "2": 8,
      "3": 45,
      "4": 9
    }
  }
```

#### SFT2 (trained_20260122_161218_relax7)
```
"metrics": {
    "accuracy": 0.78,
    "avg_steps": 8.36,
    "avg_reward": 0.6738532380834948,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 18,
      "4": 4
    }
  }
```

#### SFT3 (trained_20260122_161545_relax7)
```
"metrics": {
    "accuracy": 0.82,
    "avg_steps": 8.5,
    "avg_reward": 0.7312657268253419,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 16,
      "4": 2
    }
  }
```

<!-- #### SFT4 (trained_20260122_161641_relax7)
```
"metrics": {
    "accuracy": 0.62,
    "avg_steps": 6.08,
    "avg_reward": 0.43378230307044696,
    "failed_breakdown": {
      "1": 2,
      "2": 0,
      "3": 36,
      "4": 0
    }
  }
``` -->

### Ablation-1000

#### Qwen3-VL-4B (holo2_screenspot_20260122_183855)
```
"metrics": {
    "success": 89,
    "total": 100,
    "failed_json": 0,
    "total_reward": 89.0
  }
```

#### Qwen3-VL-8B (grounding_baseline_20260122_183351_relax7)
```
"metrics": {
    "accuracy": 0.9,
    "avg_steps": 1.04,
    "avg_reward": 0.8363254705726432,
    "failed_breakdown": {
      "1": 3,
      "2": 0,
      "3": 5,
      "4": 2
    }
  }
```

#### Qwen3-VL-235B-A22B (grounding_baseline_20260122_182214_relax7)
```
"metrics": {
    "accuracy": 0.89,
    "avg_steps": 1.0,
    "avg_reward": 0.8298523534695726,
    "failed_breakdown": {
      "1": 2,
      "2": 0,
      "3": 8,
      "4": 1
    }
  }
```

#### Qwen3-VL-235B-A22B-PathFinding (api_baseline_20260122_191429_relax7)

#### SFT2 (trained_20260122_181943_relax7)
```
"metrics": {
    "accuracy": 0.68,
    "avg_steps": 9.47,
    "avg_reward": 0.532860897966668,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 27,
      "4": 5
    }
  }
```

#### SFT3 (trained_20260122_182059_relax7)
```
"metrics": {
    "accuracy": 0.79,
    "avg_steps": 8.57,
    "avg_reward": 0.6935299232308542,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 16,
      "4": 5
    }
  }
```

#### SFT4 ()

### Ablation-800

#### Qwen3-VL-4B (holo2_screenspot_20260122_190227)
```
"metrics": {
    "success": 91,
    "total": 100,
    "failed_json": 0,
    "total_reward": 91.0
  }
```

#### Qwen3-VL-8B ()

#### Qwen3-VL-235B-A22B ()

#### Qwen3-VL-235B-A22B-PathFinding ()

#### SFT2 ()
```
```

#### SFT3 ()

#### SFT4 ()

### Ablation-600

#### Qwen3-VL-4B (holo2_screenspot_20260122_194933)
```
"metrics": {
    "success": 83,
    "total": 100,
    "failed_json": 2,
    "total_reward": 83.0
  }
```

#### Qwen3-VL-8B ()

#### Qwen3-VL-235B-A22B ()

#### Qwen3-VL-235B-A22B-PathFinding ()

#### SFT2 (trained_20260122_210234_relax7)
```
"metrics": {
    "accuracy": 0.46,
    "avg_steps": 8.35,
    "avg_reward": 0.2394679085471803,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 46,
      "4": 8
    }
  }
```

#### SFT3 (trained_20260122_210303_relax7)
```
"metrics": {
    "accuracy": 0.54,
    "avg_steps": 9.67,
    "avg_reward": 0.3395796244188722,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 38,
      "4": 8
    }
  }
```

#### SFT4 ()

### Ablation-Origin

#### SFT2 (trained_20260122_190945_relax7)
```
"metrics": {
    "accuracy": 0.79,
    "avg_steps": 9.68,
    "avg_reward": 0.6770105865814671,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 20,
      "4": 1
    }
  }
```

#### SFT3 (trained_20260122_191007_relax7)
```
"metrics": {
    "accuracy": 0.84,
    "avg_steps": 11.61,
    "avg_reward": 0.7550981669442723,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 14,
      "4": 2
    }
  }
```


等待记录结果：origin复现

----------------------------
#### SFT2@ScreenSpot (trained_20260123_220451_relax7)
```
"metrics": {
    "accuracy": 0.56,
    "avg_steps": 9.66,
    "avg_reward": 0.34342285742435746,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 41,
      "4": 3
    }
  }
```

#### SFT3@ScreenSpot (trained_20260123_220553_relax7)
```
"metrics": {
    "accuracy": 0.53,
    "avg_steps": 11.75,
    "avg_reward": 0.30479688501184454,
    "failed_breakdown": {
      "1": 0,
      "2": 0,
      "3": 42,
      "4": 5
    }
  }
```