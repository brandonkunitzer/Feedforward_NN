# Feedforward Neural Network
## Neural Network used to predict fantasy football outcomes

Have you ever wanted the edge in a fantasy football match? I did, that's why I set forth on the journey of making this project. The goal of this project is to use a neural network to predict whether or not a fantasy football player will outperform their fantasy football projections. However, I decided to challenge myself a little bit more and developed the neural network myself. I developed a feedforward neural network that uses sigmoid activation to classify whether or not a player is going to outperform their projection. With this project I included:

* Code for the neural network
* The data I used to train the model
* A saved version of the model and scaler for data - and a file to run this model

# Project in action
<img width="308" alt="Screenshot 2024-08-24 at 10 58 12 PM" src="https://github.com/user-attachments/assets/749d4399-b1d6-4f09-982b-78c5ac7c3ef0">

# Neural Network Details
![IMG_0011](https://github.com/user-attachments/assets/53dc32e2-33c8-4bde-8fcc-599c85295d41)
This project uses a neural network that takes in four inputs and has one hidden layer with 4 neruons. The 4 inputs are floats that are scaled using a standard scaler. As described in the intro, this is a feedforward nerual network that uses sigmoid to classify a player's performance. I used numpy to develop the neural network. 

# Fantsay Football Data Aquisition
Using selenium and beautiful soup, I was able to scrape the fantasy football data from numerous different websites

# How to use the saved model:

Step 1) Download the model and file to run the model <br />
Step 2) Open the file in a code editor <br />
Step 3) Run the file <br />
Step 4) Enter predicitons <br />
Step 5) Analyze the predicitons: if it is less than .5, then the player is not predicted to outperform their projections. If the prediction is greater than .5 then they are projected to outperform their projection.

# DISCLAIMER
This model was only traied on running back data, therefore it should only be used to predict running back performances! Also, the data used to train the model is not alwasy indicative of how a player will play. There was an in-sample accuracy of only 70% at best. Additionally, the model is most accurate for players predicted under 5 or 6 points. Anything above that and the model will disproportionately predict the player to underperform. With all of that said, the neural network works exactly as it is supposed to. The sometimes poor results are likely a consquence of the inputs having minimal inherent correlation to a player's projections. 



