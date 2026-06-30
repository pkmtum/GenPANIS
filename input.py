
cudaIndex = 0
device='cuda'
dataType = 'float'
nele = 32
shapeFuncsDim = 16
reducedDim = 17
createNewCondField = False
saveDatasetOption = False
importDatasetOption = True
saveModelOption = False
outOfDistributionPrediction = False
numOfTestSamples = 100
options={'modeType': 'test' + str(numOfTestSamples),
  'volumeFraction': 'FR50',
  'lengthScale': 0.25,
  'inputDimensions': 512,
  'boundaryCondition': 0., # 1. for training Helmholz
  'pde_type': 'darcy',  # 'darcy' or 'helmholz'
  'alpha': 0.0000,
  'u0': 5.,
  'integrationGrid': 129,
  'contrastRatio': 'CR10',
  'volumeFractionOutOfDistribution': 'FR50',
  'refSolverIntGrid': 128}
compareWithCGProjection = False
loadProjections = False
yFMode=False
Nx_samp = 10
randResBatchSize = 400
Navg = 100
mean_px = 0
sigma_px = 1
IterSvi = 50
schedulerIter = 10
lr = 0.001
gradLr = 0.1
Iter_grad = 1
eigRelax = None
sigma_rExponent = 3.5 #3.5 
stdInit = -2  # 3, latest -2
sigma_r = 10 ** (-sigma_rExponent)
sigma_w = 10**(8) ## It should be 10**8
externalIter = 1
poly_pow = 4
powerIterTol = 10 ** (-5)
movingAvgNum = 100
convTol = 20
display_plots = False
readFromCheckPoint = False
rhs = -100.
saveFolder = './DarcyFlowData/darcy_'
if isinstance(options['boundaryCondition'], str):
  saveDatasetName = saveFolder+options['modeType']+'_pwcDimX'+str(options['inputDimensions'])+'RefIntGrid'+str(options['refSolverIntGrid']) + \
  options['contrastRatio']+options['volumeFraction']+'D'+options['boundaryCondition']+'int'+str(options['integrationGrid'])+'l'+f"{options['lengthScale']:1.2f}"+'.pt'
  saveDatasetNameOutOfDist = saveFolder+options['modeType']+'_pwcDimX'+str(options['inputDimensions'])+ 'RefIntGrid'+str(options['refSolverIntGrid']) +\
options['contrastRatio']+options['volumeFractionOutOfDistribution']+'D'+options['boundaryCondition']+'int'+str(options['integrationGrid'])+'l'+f"{options['lengthScale']:1.2f}"+'.pt'
else:
  saveDatasetName = saveFolder+options['modeType']+'_pwcDimX'+str(options['inputDimensions'])+'RefIntGrid'+str(options['refSolverIntGrid']) +\
  options['contrastRatio']+options['volumeFraction']+'D'+str(int(options['boundaryCondition']))+'int'+str(options['integrationGrid'])+'l'+f"{options['lengthScale']:1.2f}"+'.pt'
  saveDatasetNameOutOfDist = saveFolder+options['modeType']+'_pwcDimX'+str(options['inputDimensions']) +'RefIntGrid'+str(options['refSolverIntGrid']) +\
options['contrastRatio']+options['volumeFractionOutOfDistribution']+'D'+str(int(options['boundaryCondition']))+'int'+str(options['integrationGrid'])+'l'+f"{options['lengthScale']:1.2f}"+'.pt'
