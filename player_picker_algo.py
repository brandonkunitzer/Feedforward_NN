import joblib

#load the model
model, scaler = joblib.load('Downloads/RB_trained_model.pkl')

#inputs from the user
Proj = input('Players projected points: ')
QB_ADP = input('Players quarterbacks ADP: ')
OPP_ADP = input('Opponents Defense ADP: ')
ADP = input('Players ADP (Average Draft Position): ')

#convert inputs into a data point to feedforward
player = [float(Proj), float(QB_ADP), float(OPP_ADP), float(ADP)]

#scale the data
player_scaled = scaler.transform([player])

#make prediction and print to terminal
prediction = model.feedforward(player_scaled[0])  

#if the prediciton is above .5, the player is likely to outperform
#their projection, and if it is below .5, the player is unlikely to 
# outperform their projection
print(prediction)