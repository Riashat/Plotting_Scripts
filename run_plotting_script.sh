
for dir in ./*/
do
 dir=${dir%*/}
  python /Users/breakend/Documents/code/machine_learning/ReproducibilityInContinuousPolicyGradientMethods/ave_results.py ${dir##*/}/exp_1/progress.csv ${dir##*/}/exp_2/progress.csv ${dir##*/}/exp_3/progress.csv ${dir##*/}/exp_4/progress.csv ${dir##*/}/exp_5/progress.csv ${dir##*/}/average.csv
  python /Users/breakend/Documents/code/machine_learning/ReproducibilityInContinuousPolicyGradientMethods/std_error.py ${dir##*/}/exp_1/progress.csv ${dir##*/}/exp_2/progress.csv ${dir##*/}/exp_3/progress.csv ${dir##*/}/exp_4/progress.csv ${dir##*/}/exp_5/progress.csv ${dir##*/}/sem.csv
  python /Users/breakend/Documents/code/machine_learning/ReproducibilityInContinuousPolicyGradientMethods/combine_std_ave.py ${dir##*/}/average.csv ${dir##*/}/sem.csv
done