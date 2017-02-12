

import model_validation.Validate as Validate
from Datasets import SimulatedData,Gjoa,GJOA2
#import Models.NeuralNetworks.NET1 as NN1
from sklearn.preprocessing import StandardScaler


if __name__=='__main__':

    GjoaData=GJOA2.fetch_gjoa_data()


    #SimData = SimulatedData.fetchSimulatedData()

    #print('\n \n')


    #print(chkInputs.columns)

    Validate.validate_train_test_split(GjoaData)

    #Validate.validateRepeat(GjoaData)
    #Validate.validateCV(GjoaData)
    #NN1.SSNET1()