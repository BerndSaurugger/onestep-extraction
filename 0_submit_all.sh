#!/bin/bash

jid=$(sbatch --parsable 1_attack.sbatch)
jid=$(sbatch --parsable --dependency=afterok:$jid 2_compare.sbatch)
jid=$(sbatch --parsable --dependency=afterok:$jid 3_multiple.sbatch)
jid=$(sbatch --parsable --dependency=afterok:$jid 4_compare.sbatch)