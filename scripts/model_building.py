from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib




def train_model(algorithm,data,x_train,y_train,x_test,y_test):
    model=algorithm
    model=model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    return (f'{data} accuracy in {algorithm} is : {accuracy_score(y_test,y_pred,)}')



def visualize(x_train, y_train,y_test, x_test,name):
    
    model = RandomForestClassifier(class_weight='balanced')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(x_test, y_pred, color='blue', label='Predictions', alpha=0.5)
    plt.scatter(x_test, y_test[:len(x_test)], color='orange', label='True Values', alpha=0.5)
    plt.title('Random Forest Predictions vs True Values for'+''+name)
    plt.xlabel('Test Data Features')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    


def visualize_random_forest_predictions(X_train_fraud, y_train_fraud, X_test_fraud, y_test_fraud,
                                         X_train_creditcard, y_train_creditcard, X_test_creditcard, y_test_creditcard):
    
    model_fraud = RandomForestClassifier(class_weight='balanced')
    model_fraud.fit(X_train_fraud, y_train_fraud)
    y_pred_fraud = model_fraud.predict(X_test_fraud)

    
    accuracy_fraud = accuracy_score(y_test_fraud, y_pred_fraud)
    print(f"Accuracy for Fraud Data: {accuracy_fraud:.2f}")

   
    model_creditcard = RandomForestClassifier(class_weight='balanced')
    model_creditcard.fit(X_train_creditcard, y_train_creditcard)
    y_pred_creditcard = model_creditcard.predict(X_test_creditcard)

   
    accuracy_creditcard = accuracy_score(y_test_creditcard, y_pred_creditcard)
    print(f"Accuracy for Credit Card Data: {accuracy_creditcard:.2f}")

   
    plt.figure(figsize=(12, 6))

  
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_fraud, y_pred_fraud, color='blue', alpha=0.5)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--') 
    plt.title('Random Forest Predictions vs True Values for Fraud Data')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid()
    plt.xlim(-0.1, 1.1) 
    plt.ylim(-0.1, 1.1)

    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test_creditcard, y_pred_creditcard, color='black', alpha=0.5)
    plt.plot([0, 1], [0, 1], color='yellow', linestyle='--') 
    plt.title('Random Forest Predictions vs True Values for Credit Card Data')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid()
    plt.xlim(-0.1, 1.1)  
    plt.ylim(-0.1, 1.1)

   
    plt.tight_layout()
    plt.show()



def save_the_model(X_train_fraud, y_train_fraud, X_train_credit, y_train_credit, output_dir):
    try:
       
        model_fraud = RandomForestClassifier(class_weight='balanced')
        model_fraud.fit(X_train_fraud, y_train_fraud)

       
        model_credit = RandomForestClassifier(class_weight='balanced')
        model_credit.fit(X_train_credit, y_train_credit)

     
        joblib.dump(model_fraud, f'{output_dir}/model_fraud.pkl')
        joblib.dump(model_credit, f'{output_dir}/model_credit.pkl')
        
        print("Models saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving the models: {e}")


