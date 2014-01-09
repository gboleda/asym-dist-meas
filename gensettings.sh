#!/bin/sh

# crossval settings

#for model in svm lsvm logreg; do
for model in svm lsvm logreg; do
    #for features in normdiffs normvectors diffs vectors; do
    for features in normdiffs vectors; do
        for space in /scratch/cluster/roller/CORE_SS.vectorspace.ppmi.svd_500.pkl /scratch/cluster/roller/mikolov.pkl /scratch/cluster/roller/CORE_SS.window2.vectorspace.ppmi.svd_500.pkl /scratch/cluster/roller/CORE_SS.typedm.svd_300.small.pkl; do
            export spacename="`basename ${space}`"
            export outdir="output/crossval/${spacename}/${features}/${model}"
            echo mkdir -p $outdir
            #export data="kotlermann_judgements.txt"; export target="entails"
            #echo condorizer python $PWD/bettersvm.py crossval -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} $outdir/$data.log
            export data="noun-noun-entailment-dataset-baroni-etal-eacl2012.txt"; export target="entails"
            echo condorizer python $PWD/bettersvm.py crossval -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} $outdir/$data.log
            export data="smallBLESS.txt"; export target="relation"
            echo condorizer python $PWD/bettersvm.py crossval -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} $outdir/$data.log
        done
    done
done

# stratify by word1
# for model in svm lsvm logreg; do
#     for features in normdiffs normvectors diffs vectors; do
# #        for space in /scratch/cluster/roller/CORE_SS.vectorspace.ppmi.svd_500.pkl /scratch/cluster/roller/mikolov.pkl; do
#         for space in /scratch/cluster/roller/CORE_SS.window2.vectorspace.ppmi.svd_500.pkl /scratch/cluster/roller/CORE_SS.typedm.svd_300.small.pkl; do
#             export spacename="`basename ${space}`"
#             export strat="word1"
#             export outdir="output/unseen/${strat}/${spacename}/${features}/${model}"
#             echo mkdir -p $outdir
#             export data="kotlermann_judgements.txt"; export target="entails"
#             echo condorizer python $PWD/bettersvm.py unseen -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} --stratifier ${strat} $outdir/$data.log
#             export data="noun-noun-entailment-dataset-baroni-etal-eacl2012.txt"; export target="entails"
#             echo condorizer python $PWD/bettersvm.py unseen -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} --stratifier ${strat} $outdir/$data.log
#             export data="smallBLESS.txt"; export target="relation"
#             echo condorizer python $PWD/bettersvm.py unseen -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} --stratifier ${strat} $outdir/$data.log
# 
#             # and stratify by the category for bless
#             export strat="category"
#             export outdir="output/unseen/${strat}/${spacename}/${features}/${model}"
#             echo mkdir -p $outdir
#             export data="smallBLESS.txt"; export target="relation"
#             echo condorizer python $PWD/bettersvm.py unseen -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} --stratifier ${strat} $outdir/$data.log
#         done
#     done
# done

# cheating stuff
# export target="relation"
# export data="smallBLESS.txt"
# for spacename in CORE_SS.window2.vectorspace.ppmi.svd_500.pkl CORE_SS.typedm.svd_300.small.pkl mikolov.pkl CORE_SS.vectorspace.ppmi.svd_500.pkl; do
#     export space="/scratch/cluster/roller/${spacename}"
#     for strat in word1 category; do
#         #for cf in 0.0 .0001 .001 .1 1.0; do
#         for cf in .0025 .005 .025 .05 ; do
#             export model="svm"
#             export features="vectors"
#             export outdir="output/cheating/${strat}/${spacename}/${features}/${model}"
# 
#             echo mkdir -p $outdir
#             echo condorizer python $PWD/bettersvm.py unseen -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} --stratifier ${strat} --cheatingfactor ${cf} $outdir/${data}-cf_${cf}.log
# 
#             export model="logreg"
#             export features="normdiffs"
#             export outdir="output/cheating/${strat}/${spacename}/${features}/${model}"
#             echo mkdir -p $outdir
# 
#             echo condorizer python $PWD/bettersvm.py unseen -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} --stratifier ${strat} --cheatingfactor ${cf} $outdir/${data}-cf_${cf}.log
#         done
#     done
# done
# 
# export target="entails"
# export data="noun-noun-entailment-dataset-baroni-etal-eacl2012.txt"
# for spacename in CORE_SS.window2.vectorspace.ppmi.svd_500.pkl CORE_SS.typedm.svd_300.small.pkl mikolov.pkl CORE_SS.vectorspace.ppmi.svd_500.pkl; do
#     export space="/scratch/cluster/roller/${spacename}"
#     export strat=word1
#     #for cf in 0.0 .001 .01 .1 1.0; do
#     for cf in .0025 .005 .025 .05 ; do
#         export model="svm"
#         export features="vectors"
#         export outdir="output/cheating/${strat}/${spacename}/${features}/${model}"
# 
#         echo mkdir -p $outdir
#         echo condorizer python $PWD/bettersvm.py unseen -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} --stratifier ${strat} --cheatingfactor ${cf} $outdir/${data}-cf_${cf}.log
# 
#         export model="logreg"
#         export features="normdiffs"
#         export outdir="output/cheating/${strat}/${spacename}/${features}/${model}"
#         echo mkdir -p $outdir
# 
#         echo condorizer python $PWD/bettersvm.py unseen -d $PWD/data/${data} -s $space -n 300 -f ${features} -m ${model} -t ${target} --stratifier ${strat} --cheatingfactor ${cf} $outdir/${data}-cf_${cf}.log
#     done
# done
# 
