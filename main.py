

import Validate
import DataManager.Simulated.DataManager as DM
#import Models.NeuralNetworks.NET1 as NN1


if __name__=='__main__':
    chkInputs, wellOutputs, totalOutput = DM.getSimulatedOilFieldData()
    Validate.validate_train_test_split(chkInputs, wellOutputs, totalOutput)
    #Validate.validateCV(chkInputs,wellOutputs,totalOutput)
    #NN1.SSNET1()