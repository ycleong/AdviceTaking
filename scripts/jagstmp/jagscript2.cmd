load dic
model in "/Users/yuanchangleong/Dropbox/Projects/FACT/AdviceTaking/scripts/ModelComparison/mbe_1gr_example.txt"
data in jagsdata.R
compile, nchains(1)
parameters in jagsinit2.R
initialize
update 1000
monitor set mu, thin(1)
monitor set sigma, thin(1)
monitor set nu, thin(1)
monitor deviance
update 30000
coda *, stem('CODA2')
