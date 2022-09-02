import os
import itertools

# No.samples:      1,     3,     7,   15,   30,   75, 150, 300, 749, 1498        (sample based subset selection)
# No.patients:     0,     0,     0,    0,    1,    4,   8,  16,  40,  80         (patient based subset selection)
subset_ratios=(0.001, 0.002, 0.005, 0.01, 0.02, 
               0.05, 0.1, 0.2, 0.5, 1.0
               )
encoders=(
          "imagenet",                                 # -> Supervised Imagenet
          "resnet50_byol_imagenet2012.pth.tar",       # -> BYOL Imagenet
          "byol_acdc_backbone_last.pth",              # -> BYOL ACDC
          "byol-imagenet-acdc-ep=34.ckpt",            # -> BYOL Imagenet + BYOL ACDC
          "byol-imagenet-acdc-ep=25.pth",             # -> BYOL Imagenet + BYOL ACDC
          "supervised-imagenet-byol-acdc-ep=25.pth",  # -> Supervised Imagenet + BYOL ACDC
          )
seeds=(0, 13, 42, 
       123, 1111,
       77, 62, 91, 7, 5 
       # 18
       )

combinations = list(itertools.product(subset_ratios, encoders, seeds))

# for i, element in enumerate(combinations):
#     print(i, ":", element)

offset = 0
run_id = int(os.environ['SLURM_ARRAY_TASK_ID']) - offset
run_config = combinations[run_id]
run_args= "--subset_ratio {} --model.encoder_weights {} --seed {}".format(*run_config)
experiment_name= " --experiment_name Swp2_Subset{}_{}_seed{}".format(run_config[0], 
                                                                    run_config[1].split('/')[-1],
                                                                    run_config[2])
run_args += experiment_name

print(run_args) #Only print the config as the output of this script 
# https://stackoverflow.com/questions/34171568/return-value-from-python-script-to-shell-script
