import pandas as pd
import numpy as np

def generate_mock_data(n_rows=100):
    """Génère un faux dataset pour tester le modèle."""
    np.random.seed(42) # Pour la reproductibilité
    
    data = {
        'Joueur': [f'Joueur_{i}' for i in range(1, n_rows + 1)],
        'Points_Journee_Precedente': np.random.randint(0, 15, size=n_rows).astype(float),
        'FDR_Adversaire': np.random.randint(1, 6, size=n_rows).astype(float), # Difficulté de 1 à 5
        'Ecart_Average': np.random.normal(0, 2, size=n_rows),
        'xG': np.random.uniform(0, 1.5, size=n_rows),
        'xA': np.random.uniform(0, 1.5, size=n_rows),
    }
    
    # Création d'une relation linéaire avec un peu de bruit pour les Points_Cibles
    # La formule simulée : Y = 2 + 0.5*Pts_JP - 1.2*FDR + 0.8*Ecart + 4*xG + 3*xA + epsilon
    data['Points_Cibles'] = (
        2 + 
        0.5 * data['Points_Journee_Precedente'] - 
        1.2 * data['FDR_Adversaire'] + 
        0.8 * data['Ecart_Average'] + 
        4.0 * data['xG'] + 
        3.0 * data['xA'] + 
        np.random.normal(0, 1, size=n_rows) # Bruit (epsilon)
    )
    
    df = pd.DataFrame(data)
    
    # Introduction de quelques 'blanks' (valeurs nulles ou matchs reportés) pour tester le nettoyage
    df.loc[5:10, 'Points_Journee_Precedente'] = np.nan
    df.loc[20:25, 'xG'] = np.nan
    
    return df

def clean_data(df):
    """Nettoie les données en retirant les lignes avec des valeurs nulles (blanks)."""
    print(f"Nombre de lignes avant nettoyage : {len(df)}")
    # dropna() supprime toute ligne contenant au moins un NaN
    df_cleaned = df.dropna()
    print(f"Nombre de lignes après nettoyage : {len(df_cleaned)}")
    return df_cleaned

def calculate_beta(X, y):
    """
    Calcule l'estimateur des moindres carrés ordinaires (MCO) : 
    Beta_hat = (X^T * X)^-1 * X^T * y
    """
    # Ajouter une colonne de 1 pour l'ordonnée à l'origine (intercept)
    # X devient une matrice de taille (n_samples, n_features + 1)
    X_b = np.c_[np.ones((len(X), 1)), X]
    
    # Calcul matriciel
    X_T = X_b.T
    # np.linalg.inv calcule l'inverse de la matrice
    # .dot() effectue la multipl    ication matricielle
    beta_hat = np.linalg.inv(X_T.dot(X_b)).dot(X_T).dot(y)
    
    return beta_hat

def predict(X, beta_hat):
    """Calcule les prédictions Y_hat = X * beta_hat"""
    X_b = np.c_[np.ones((len(X), 1)), X]
    return X_b.dot(beta_hat)

def main():
    print("=== Prédiction des points FPL par Régression Linéaire Multiple ===\n")
    
    # 1. Génération des données
    print("--- 1. Génération des données mock ---")
    df = generate_mock_data(100)
    
    # 2. Nettoyage des données
    print("\n--- 2. Nettoyage des données ---")
    df_clean = clean_data(df)
    
    # 3. Préparation des variables
    # X contient les variables explicatives, y contient la variable à prédire
    features = ['Points_Journee_Precedente', 'FDR_Adversaire', 'Ecart_Average', 'xG', 'xA']
    X = df_clean[features].values
    y = df_clean['Points_Cibles'].values
    joueurs = df_clean['Joueur'].values
    
    # 4. Calcul de l'estimateur Beta
    print("\n--- 3. Entraînement du modèle (Calcul de Beta_hat) ---")
    beta_hat = calculate_beta(X, y)
    
    # Affichage des coefficients
    print("Coefficients calculés (\u03b2) :")
    print(f"  Intercept (\u03b20) : {beta_hat[0]:.4f}")
    for i, feature in enumerate(features):
        print(f"  {feature} (\u03b2{i+1}) : {beta_hat[i+1]:.4f}")
        
    # 5. Prédictions
    print("\n--- 4. Prédictions sur un échantillon ---")
    y_pred = predict(X, beta_hat)
    
    print(f"{'Joueur':<15} | {'Points Cibles (Réels)':<25} | {'Points Prédits':<20}")
    print("-" * 65)
    for i in range(min(10, len(y_pred))):
        print(f"{joueurs[i]:<15} | {y[i]:<25.2f} | {y_pred[i]:<20.2f}")

if __name__ == "__main__":
    main()
