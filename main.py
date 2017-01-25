

import model_validation.Validate as Validate
from Datasets import SimulatedData,Gjoa
#import Models.NeuralNetworks.NET1 as NN1
from sklearn.preprocessing import StandardScaler


if __name__=='__main__':

    GjoaData=Gjoa.fetch_gjoa_data()


    SimData = SimulatedData.fetchSimulatedData()

    print('\n \n')


    #print(chkInputs.columns)

    Validate.validate_train_test_split(GjoaData)
    #Validate.validateCV(chkInputs,wellOutputs,totalOutput)
    #NN1.SSNET1()