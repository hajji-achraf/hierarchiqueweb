import streamlit as st
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt


# Fonction pour calculer la distance euclidienne entre deux points
def distance_euclidienne(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Fonction pour calculer la matrice de distances euclidiennes entre les points
def calculer_matrice_distances(points):
    num_points = len(points)
    distances = np.zeros((num_points, num_points))  # Initialisation de la matrice de distances

    for i in range(num_points):
        for j in range(num_points):
            distances[i, j] = distance_euclidienne(points[i], points[j])

    return distances


# Fonction pour calculer le barycentre d'un groupe de points
def calculer_barycentre_groupe(points_groupe, noms_groupe):
    barycentre = np.mean(points_groupe, axis=0)
    return barycentre, noms_groupe


# Fonction pour afficher les données d'exemple
def afficher_donnees_exemple():
    st.subheader("Données d'exemple")
    exemple_data = {
        "Feature 1": [8.0, 2.0, 3.0, 1.0, 1.0],
        "Feature 2": [2.0, 2.0, 4.0, 2.0, 6.0],
        "Feature 3": [3.0, 3.0, 5.0, 6.0, 7.0]
    }
    df_exemple = pd.DataFrame(exemple_data)
    df_exemple.index += 1  # Modifie l'index pour commencer à 1
    st.write("Voici un exemple de données :")
    st.dataframe(df_exemple)
    return df_exemple


# Fonction pour ajouter du CSS à l'application
def ajouter_css():
    st.markdown("""
    <style>
    .title {
        color: #4CAF50;
        font-size: 2em;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f5;
    }
    .sidebar .sidebar-content .element-container {
        color: #333;
    }
    .stDataFrame {
        border: 2px solid #4CAF50;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# Définition de l'application Streamlit
def main():
    # Ajouter le CSS
    ajouter_css()

    st.markdown("<h1 class='title'>Analyse de Clustering Hiérarchique</h1>", unsafe_allow_html=True)
    st.markdown("[by Achraf Hajji](https://www.linkedin.com/in/achraf-hajji-a271ab241/)")

    # Sidebar pour importer les données
    st.sidebar.header("Importer les Données")
    uploaded_file = st.sidebar.file_uploader("Importer un fichier Excel contenant les points", type=["xlsx", "xls"])

    # Sidebar pour tester l'algorithme
    st.sidebar.header("Tester l'Algorithme")
    linkage_methods = ['single', 'complete', 'average', 'ward']
    selected_linkage = st.sidebar.selectbox("Choisir la méthode de liaison :", linkage_methods)

    # Afficher les données d'exemple
    st.sidebar.header("Afficher les Données d'Exemple")
    if st.sidebar.button("Afficher les Données d'Exemple"):
        example_data = afficher_donnees_exemple()

        # Calculer la matrice de distance pour le dendrogramme
        distances = hierarchy.distance.pdist(example_data.values)

        # Effectuer la classification hiérarchique pour le dendrogramme
        linkage_matrix = hierarchy.linkage(distances, method=selected_linkage)

        # Tracer le dendrogramme
        fig, ax = plt.subplots(figsize=(10, 5))
        hierarchy.dendrogram(linkage_matrix, labels=example_data.index.astype(str), ax=ax)
        ax.set_xlabel('Indices des données')
        ax.set_ylabel('Distance')
        ax.set_title('Dendrogramme de Classification Hiérarchique pour Données d\'Exemple')

        # Afficher le dendrogramme dans l'interface Streamlit
        st.pyplot(fig)

    # Traitement des données importées
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        points = data.values
        data.index += 1

        # Afficher les données importées
        st.write("Données importées :")
        st.dataframe(data)

        # Calcul du regroupement hiérarchique
        points_restants = points.copy()
        noms_restants = [f"Point {i + 1}" for i in range(len(points))]

        while len(points_restants) > 1:
            matrice_distances_restants = calculer_matrice_distances(points_restants)
            np.fill_diagonal(matrice_distances_restants, np.inf)
            indice_min = np.unravel_index(np.argmin(matrice_distances_restants, axis=None),
                                          matrice_distances_restants.shape)

            point_min1 = points_restants[indice_min[0]]
            point_min2 = points_restants[indice_min[1]]
            nom_min1 = noms_restants[indice_min[0]]
            nom_min2 = noms_restants[indice_min[1]]

            barycentre, noms_regroupes = calculer_barycentre_groupe([point_min1, point_min2], [nom_min1, nom_min2])

            st.write(f"Points regroupés : {noms_regroupes}")
            st.write(f"Distance minimale : {matrice_distances_restants[indice_min]}")

            points_restants = np.vstack(
                [barycentre, np.delete(points_restants, [indice_min[0], indice_min[1]], axis=0)])
            noms_restants = [noms_regroupes] + [nom for i, nom in enumerate(noms_restants) if i not in indice_min]

        st.write("Barycentre final :", barycentre)

        # Calculer la matrice de distance pour le dendrogramme
        distances = hierarchy.distance.pdist(points)

        # Effectuer la classification hiérarchique pour le dendrogramme
        linkage_matrix = hierarchy.linkage(distances, method=selected_linkage)

        # Tracer le dendrogramme
        fig, ax = plt.subplots(figsize=(10, 5))
        hierarchy.dendrogram(linkage_matrix, labels=data.index.astype(str), ax=ax)
        ax.set_xlabel('Indices des données')
        ax.set_ylabel('Distance')
        ax.set_title('Dendrogramme de Classification Hiérarchique')

        # Afficher le dendrogramme dans l'interface Streamlit
        st.pyplot(fig)


# Lancement de l'application
if __name__ == "__main__":
    main()
