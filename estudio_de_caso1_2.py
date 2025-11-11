# -*- coding: utf-8 -*-
"""
Estudio de Caso: Análisis de Clúster - Banca y Hotelería
Universidad LEAD - BCD-6210 Minería de Datos
III Cuatrimestre 2025

Autores:
- Jason Jesús Barrantes Sánchez (jason.barrantes@ulead.ac.cr)
- Melany Ramírez Anchía (melany.ramirez@ulead.ac.cr)

Instructor: Dr. Juan Murillo Morera (juan.murillo.morera@ulead.ac.cr)
Fecha de entrega: 15 de Noviembre, 2025

Descripción:
Este script implementa un análisis completo de clustering aplicado a datos
del sector bancario (deserción de clientes) y hotelero (patrones de reserva).
Se implementan múltiples técnicas: PCA, HAC, K-means, t-SNE y UMAP.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import umap.umap_ as umap
import warnings

warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AnalisisCluster:
    """
    Clase para realizar análisis de clustering completo
    """

    def __init__(self, datos, nombre_dataset):
        self.datos = datos
        self.nombre = nombre_dataset
        self.datos_procesados = None
        self.scaler = StandardScaler()

    def preprocesar_datos(self):
        """Preprocesamiento: codificación y estandarización"""
        print(f"\n{'=' * 60}")
        print(f"PREPROCESAMIENTO: {self.nombre}")
        print(f"{'=' * 60}")

        df = self.datos.copy()

        # Separar numéricas y categóricas
        numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        categoricas = df.select_dtypes(include=['object']).columns.tolist()

        print(f"Variables numéricas: {len(numericas)}")
        print(f"Variables categóricas: {len(categoricas)}")

        # Codificar categóricas
        le = LabelEncoder()
        for col in categoricas:
            df[col] = le.fit_transform(df[col].astype(str))

        # Estandarizar
        self.datos_procesados = pd.DataFrame(
            self.scaler.fit_transform(df),
            columns=df.columns
        )

        print(f"Dimensiones finales: {self.datos_procesados.shape}")
        return self.datos_procesados

    def analisis_pca(self, n_components_list=[2, 3, 5]):
        """Análisis de Componentes Principales con múltiples configuraciones"""
        print(f"\n{'=' * 60}")
        print(f"ANÁLISIS PCA: {self.nombre}")
        print(f"{'=' * 60}")

        resultados = {}

        for n_comp in n_components_list:
            pca = PCA(n_components=n_comp)
            componentes = pca.fit_transform(self.datos_procesados)

            var_explicada = pca.explained_variance_ratio_
            var_acumulada = np.cumsum(var_explicada)

            resultados[n_comp] = {
                'pca': pca,
                'componentes': componentes,
                'varianza_explicada': var_explicada,
                'varianza_acumulada': var_acumulada
            }

            print(f"\nConfiguración: {n_comp} componentes")
            print(f"Varianza explicada por componente: {var_explicada}")
            print(f"Varianza acumulada: {var_acumulada[-1]:.4f}")

        # Visualización
        self._plot_pca_variance(resultados)
        self._plot_pca_2d(resultados[2]['componentes'])

        return resultados

    def _plot_pca_variance(self, resultados):
        """Gráfico de varianza explicada"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scree plot
        max_comp = max(resultados.keys())
        var_exp = resultados[max_comp]['varianza_explicada']
        axes[0].bar(range(1, len(var_exp) + 1), var_exp)
        axes[0].set_xlabel('Componente Principal')
        axes[0].set_ylabel('Varianza Explicada')
        axes[0].set_title(f'Scree Plot - {self.nombre}')

        # Varianza acumulada
        var_acum = resultados[max_comp]['varianza_acumulada']
        axes[1].plot(range(1, len(var_acum) + 1), var_acum, 'bo-')
        axes[1].axhline(y=0.9, color='r', linestyle='--', label='90%')
        axes[1].set_xlabel('Número de Componentes')
        axes[1].set_ylabel('Varianza Acumulada')
        axes[1].set_title(f'Varianza Acumulada - {self.nombre}')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'pca_variance_{self.nombre}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_pca_2d(self, componentes):
        """Visualización 2D de PCA"""
        plt.figure(figsize=(10, 8))
        plt.scatter(componentes[:, 0], componentes[:, 1], alpha=0.6, s=50)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Proyección PCA 2D - {self.nombre}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'pca_2d_{self.nombre}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def clustering_jerarquico(self, n_clusters_list=[2, 3, 4],
                              linkage_methods=['ward', 'complete', 'average']):
        """Clustering Jerárquico con múltiples configuraciones"""
        print(f"\n{'=' * 60}")
        print(f"CLUSTERING JERÁRQUICO: {self.nombre}")
        print(f"{'=' * 60}")

        resultados = {}
        mejores_metricas = {'silhouette': -1, 'config': None}

        for method in linkage_methods:
            for n_clust in n_clusters_list:
                hac = AgglomerativeClustering(
                    n_clusters=n_clust,
                    linkage=method
                )
                labels = hac.fit_predict(self.datos_procesados)

                # Métricas
                sil = silhouette_score(self.datos_procesados, labels)
                db = davies_bouldin_score(self.datos_procesados, labels)
                ch = calinski_harabasz_score(self.datos_procesados, labels)

                config = f"{method}_{n_clust}"
                resultados[config] = {
                    'labels': labels,
                    'silhouette': sil,
                    'davies_bouldin': db,
                    'calinski_harabasz': ch
                }

                print(f"\n{method.upper()} - {n_clust} clusters:")
                print(f"  Silhouette: {sil:.4f}")
                print(f"  Davies-Bouldin: {db:.4f}")
                print(f"  Calinski-Harabasz: {ch:.2f}")

                if sil > mejores_metricas['silhouette']:
                    mejores_metricas = {'silhouette': sil, 'config': config}

        print(f"\n{'*' * 60}")
        print(f"MEJOR CONFIGURACIÓN: {mejores_metricas['config']}")
        print(f"Silhouette Score: {mejores_metricas['silhouette']:.4f}")
        print(f"{'*' * 60}")

        # Dendrograma para Ward
        self._plot_dendrogram()

        return resultados, mejores_metricas

    def _plot_dendrogram(self):
        """Generar dendrograma"""
        plt.figure(figsize=(12, 6))
        Z = linkage(self.datos_procesados, method='ward')
        dendrogram(Z, truncate_mode='lastp', p=30)
        plt.title(f'Dendrograma (Ward) - {self.nombre}')
        plt.xlabel('Índice de muestra o (tamaño del cluster)')
        plt.ylabel('Distancia')
        plt.savefig(f'dendrogram_{self.nombre}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def kmeans_analysis(self, k_range=range(2, 11)):
        """Análisis K-means con método del codo"""
        print(f"\n{'=' * 60}")
        print(f"ANÁLISIS K-MEANS: {self.nombre}")
        print(f"{'=' * 60}")

        inertias = []
        silhouettes = []
        resultados = {}

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
            labels = kmeans.fit_predict(self.datos_procesados)

            inertia = kmeans.inertia_
            sil = silhouette_score(self.datos_procesados, labels)

            inertias.append(inertia)
            silhouettes.append(sil)

            resultados[k] = {
                'model': kmeans,
                'labels': labels,
                'inertia': inertia,
                'silhouette': sil
            }

            print(f"k={k}: Inertia={inertia:.2f}, Silhouette={sil:.4f}")

        # Gráfico del codo y silhouette
        self._plot_elbow_silhouette(k_range, inertias, silhouettes)

        # Mejor k según silhouette
        mejor_k = k_range[np.argmax(silhouettes)]
        print(f"\nMejor k según Silhouette: {mejor_k}")

        return resultados, mejor_k

    def _plot_elbow_silhouette(self, k_range, inertias, silhouettes):
        """Gráficos de método del codo y silhouette"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Método del codo
        axes[0].plot(k_range, inertias, 'bo-')
        axes[0].set_xlabel('Número de Clusters (k)')
        axes[0].set_ylabel('Inercia')
        axes[0].set_title(f'Método del Codo - {self.nombre}')
        axes[0].grid(True)

        # Silhouette score
        axes[1].plot(k_range, silhouettes, 'ro-')
        axes[1].set_xlabel('Número de Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title(f'Silhouette Score - {self.nombre}')
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'kmeans_metrics_{self.nombre}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def tsne_analysis(self, perplexity_list=[30, 50]):
        """Análisis t-SNE con diferentes perplexities"""
        print(f"\n{'=' * 60}")
        print(f"ANÁLISIS t-SNE: {self.nombre}")
        print(f"{'=' * 60}")

        resultados = {}

        fig, axes = plt.subplots(1, len(perplexity_list), figsize=(14, 6))
        if len(perplexity_list) == 1:
            axes = [axes]

        for idx, perp in enumerate(perplexity_list):
            print(f"\nCalculando t-SNE con perplexity={perp}...")
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
            embedding = tsne.fit_transform(self.datos_procesados)

            resultados[perp] = embedding

            # Visualización
            axes[idx].scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=30)
            axes[idx].set_title(f't-SNE (perplexity={perp})')
            axes[idx].set_xlabel('Dimensión 1')
            axes[idx].set_ylabel('Dimensión 2')

        plt.tight_layout()
        plt.savefig(f'tsne_{self.nombre}.png', dpi=300, bbox_inches='tight')
        plt.show()

        return resultados

    def umap_analysis(self, n_neighbors_list=[15, 30]):
        """Análisis UMAP con diferentes n_neighbors"""
        print(f"\n{'=' * 60}")
        print(f"ANÁLISIS UMAP: {self.nombre}")
        print(f"{'=' * 60}")

        resultados = {}

        fig, axes = plt.subplots(1, len(n_neighbors_list), figsize=(14, 6))
        if len(n_neighbors_list) == 1:
            axes = [axes]

        for idx, n_neigh in enumerate(n_neighbors_list):
            print(f"\nCalculando UMAP con n_neighbors={n_neigh}...")
            reducer = umap.UMAP(n_neighbors=n_neigh, random_state=42)
            embedding = reducer.fit_transform(self.datos_procesados)

            resultados[n_neigh] = embedding

            # Visualización
            axes[idx].scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=30)
            axes[idx].set_title(f'UMAP (n_neighbors={n_neigh})')
            axes[idx].set_xlabel('Dimensión 1')
            axes[idx].set_ylabel('Dimensión 2')

        plt.tight_layout()
        plt.savefig(f'umap_{self.nombre}.png', dpi=300, bbox_inches='tight')
        plt.show()

        return resultados

    def generar_reporte_completo(self, pca_results, hac_results, kmeans_results,
                                 mejor_hac, mejor_k):
        """
        Genera un reporte completo con todos los resultados del análisis
        """
        print(f"\n{'=' * 70}")
        print(f"REPORTE COMPLETO DE RESULTADOS - {self.nombre}")
        print(f"{'=' * 70}")

        # Resumen PCA
        print("\n1. ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)")
        print("-" * 70)
        for n_comp, res in pca_results.items():
            var_total = res['varianza_acumulada'][-1]
            print(f"   • {n_comp} componentes: {var_total:.2%} varianza explicada")

        # Resumen HAC
        print("\n2. CLUSTERING JERÁRQUICO (HAC)")
        print("-" * 70)
        print(f"   • Mejor configuración: {mejor_hac['config']}")
        print(f"   • Silhouette Score: {mejor_hac['silhouette']:.4f}")

        mejor_config = hac_results[mejor_hac['config']]
        print(f"   • Davies-Bouldin: {mejor_config['davies_bouldin']:.4f}")
        print(f"   • Calinski-Harabasz: {mejor_config['calinski_harabasz']:.2f}")

        # Distribución de clusters
        labels = mejor_config['labels']
        unique, counts = np.unique(labels, return_counts=True)
        print("   • Distribución de clusters:")
        for cluster, count in zip(unique, counts):
            print(f"     - Cluster {cluster}: {count} observaciones ({count / len(labels):.1%})")

        # Resumen K-means
        print("\n3. K-MEANS CLUSTERING")
        print("-" * 70)
        print(f"   • Número óptimo de clusters: {mejor_k}")
        mejor_kmeans = kmeans_results[mejor_k]
        print(f"   • Silhouette Score: {mejor_kmeans['silhouette']:.4f}")
        print(f"   • Inercia: {mejor_kmeans['inertia']:.2f}")

        # Distribución de clusters K-means
        labels_km = mejor_kmeans['labels']
        unique_km, counts_km = np.unique(labels_km, return_counts=True)
        print("   • Distribución de clusters:")
        for cluster, count in zip(unique_km, counts_km):
            print(f"     - Cluster {cluster}: {count} observaciones ({count / len(labels_km):.1%})")

        print("\n" + "=" * 70)

        # Guardar resultados en archivo
        with open(f'reporte_{self.nombre}.txt', 'w', encoding='utf-8') as f:
            f.write(f"REPORTE DE RESULTADOS - {self.nombre}\n")
            f.write(f"Fecha: {pd.Timestamp.now()}\n")
            f.write(f"{'=' * 70}\n\n")

            f.write("MEJOR CONFIGURACIÓN HAC:\n")
            f.write(f"Config: {mejor_hac['config']}\n")
            f.write(f"Silhouette: {mejor_hac['silhouette']:.4f}\n\n")

            f.write("MEJOR CONFIGURACIÓN K-MEANS:\n")
            f.write(f"k: {mejor_k}\n")
            f.write(f"Silhouette: {mejor_kmeans['silhouette']:.4f}\n")
            f.write(f"Inercia: {mejor_kmeans['inertia']:.2f}\n")

        print(f"\n✓ Reporte guardado en: reporte_{self.nombre}.txt")


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  ESTUDIO DE CASO: ANÁLISIS DE CLÚSTER                      ║
    ║  Banca y Hotelería - Universidad LEAD                      ║
    ║  BCD-6210 Minería de Datos                                 ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Cargar datos
    print("Cargando datasets...")
    df_banca = pd.read_csv('BankChurners.csv')
    df_hotel = pd.read_csv('hotel_bookings_muestra.csv')

    # Remover columnas no necesarias
    df_banca = df_banca.drop(['ID'], axis=1, errors='ignore')
    df_hotel = df_hotel.drop(df_hotel.columns[0], axis=1, errors='ignore')

    print(f"✓ Dataset Banca: {df_banca.shape}")
    print(f"✓ Dataset Hotel: {df_hotel.shape}")

    # =========================================================================
    # ANÁLISIS DATASET BANCA
    # =========================================================================

    print("\n\n" + "█" * 60)
    print("█" + " " * 20 + "DATASET BANCA" + " " * 27 + "█")
    print("█" * 60)

    analisis_banca = AnalisisCluster(df_banca, "Banca")
    analisis_banca.preprocesar_datos()

    # PCA
    pca_banca = analisis_banca.analisis_pca(n_components_list=[2, 3, 5])

    # HAC
    hac_banca, mejor_hac_banca = analisis_banca.clustering_jerarquico(
        n_clusters_list=[2, 3, 4],
        linkage_methods=['ward', 'complete', 'average']
    )

    # K-means
    kmeans_banca, mejor_k_banca = analisis_banca.kmeans_analysis(k_range=range(2, 8))

    # t-SNE
    tsne_banca = analisis_banca.tsne_analysis(perplexity_list=[30, 50])

    # UMAP
    umap_banca = analisis_banca.umap_analysis(n_neighbors_list=[15, 30])

    # Generar reporte completo
    analisis_banca.generar_reporte_completo(
        pca_banca, hac_banca, kmeans_banca,
        mejor_hac_banca, mejor_k_banca
    )

    # =========================================================================
    # ANÁLISIS DATASET HOTEL
    # =========================================================================

    print("\n\n" + "█" * 60)
    print("█" + " " * 19 + "DATASET HOTEL" + " " * 28 + "█")
    print("█" * 60)

    analisis_hotel = AnalisisCluster(df_hotel, "Hotel")
    analisis_hotel.preprocesar_datos()

    # PCA
    pca_hotel = analisis_hotel.analisis_pca(n_components_list=[2, 3, 5])

    # HAC
    hac_hotel, mejor_hac_hotel = analisis_hotel.clustering_jerarquico(
        n_clusters_list=[2, 3, 4],
        linkage_methods=['ward', 'complete', 'average']
    )

    # K-means
    kmeans_hotel, mejor_k_hotel = analisis_hotel.kmeans_analysis(k_range=range(2, 8))

    # t-SNE
    tsne_hotel = analisis_hotel.tsne_analysis(perplexity_list=[30, 50])

    # UMAP
    umap_hotel = analisis_hotel.umap_analysis(n_neighbors_list=[15, 30])

    # Generar reporte completo
    analisis_hotel.generar_reporte_completo(
        pca_hotel, hac_hotel, kmeans_hotel,
        mejor_hac_hotel, mejor_k_hotel
    )

    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================

    print("\n\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)

    print("\nDATASET BANCA:")
    print(f"  • Mejor configuración HAC: {mejor_hac_banca['config']}")
    print(f"  • Mejor k para K-means: {mejor_k_banca}")

    print("\nDATASET HOTEL:")
    print(f"  • Mejor configuración HAC: {mejor_hac_hotel['config']}")
    print(f"  • Mejor k para K-means: {mejor_k_hotel}")

    print("\n✓ Análisis completado. Gráficos guardados en el directorio actual.")
    print("=" * 60)


# ================================================================
# === Bloques añadidos: preprocesamiento, KMeans, PCA y gráficas ===
# ================================================================
# Nota: Estos bloques NO eliminan tu lógica previa; se agregan utilidades
# para análisis y visualización adicionales. Están listos para llamarse
# desde main() o desde cualquier notebook/script.

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ------------------------------
# Configuración general
# ------------------------------
FIG_OUTDIR = Path("figures_extras")
FIG_OUTDIR.mkdir(parents=True, exist_ok=True)
MAX_ROWS_SAMPLE = 4000  # muestreo para acelerar KMeans y PCA en datasets grandes
DEFAULT_K_RANGE = range(2, 9)


# ------------------------------
# Estructuras de datos
# ------------------------------
@dataclass
class ClusteringResult:
    name: str
    n_samples: int
    n_features: int
    best_k: int
    labels: np.ndarray
    pca_components_2d: np.ndarray
    silhouette_scores: List[float]
    k_range: List[int]
    figure_paths: Dict[str, str]


# ------------------------------
# Utilidades de preprocesamiento
# ------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica categóricas (LabelEncoder) y estandariza todas las columnas numéricas.
    Incluye una heurística para remover una primera columna tipo índice.
    """
    df = df.copy()

    # Remueve columna 'ID' si existe
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Heurística: si la primera columna parece un índice incremental, la quitamos
    first = df.columns[0]
    if pd.api.types.is_integer_dtype(df[first]) and df[first].is_monotonic_increasing:
        vals = df[first].to_numpy()
        if (np.all(vals == np.arange(len(vals))) or np.all(vals == np.arange(1, len(vals) + 1))):
            df = df.drop(columns=[first])

    # Muestreo para acelerar en datasets grandes
    if len(df) > MAX_ROWS_SAMPLE:
        df = df.sample(MAX_ROWS_SAMPLE, random_state=42).reset_index(drop=True)

    # Codificación de categóricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    # Escalado
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    X = pd.DataFrame(X, columns=df.columns)
    return X


# ------------------------------
# KMeans y selección de k
# ------------------------------
def best_k_by_silhouette(X: pd.DataFrame, k_range: range = DEFAULT_K_RANGE):
    sils = []
    models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=200)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sils.append(sil)
        models[k] = (km, labels, sil)
    best_idx = int(np.argmax(sils))
    best_k = list(k_range)[best_idx]
    return best_k, sils, models


# ------------------------------
# PCA
# ------------------------------
def pca_2d(X: pd.DataFrame) -> Tuple[PCA, np.ndarray]:
    p = PCA(n_components=2, random_state=42)
    comp = p.fit_transform(X)
    return p, comp


# ------------------------------
# Funciones de guardado de gráficas
# (Usar matplotlib puro y dejar que el backend elija colores por defecto)
# ------------------------------
def save_scree_plot(pca: PCA, name: str) -> str:
    plt.figure(figsize=(7, 5))
    var = pca.explained_variance_ratio_
    xs = np.arange(1, len(var) + 1)
    plt.bar(xs, var)
    plt.plot(xs, np.cumsum(var), marker="o")
    plt.xlabel("Componente principal")
    plt.ylabel("Varianza explicada / acumulada")
    plt.title(f"Scree plot PCA – {name}")
    plt.tight_layout()
    path = FIG_OUTDIR / f"pca_scree_{name}_extra.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path)


def save_pca2d_clusters(components_2d: np.ndarray, labels: np.ndarray, name: str) -> str:
    plt.figure(figsize=(7, 6))
    plt.scatter(components_2d[:, 0], components_2d[:, 1], c=labels, s=18, alpha=0.9)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA 2D con KMeans – {name}")
    plt.tight_layout()
    path = FIG_OUTDIR / f"pca2d_kmeans_{name}_extra.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path)


def save_silhouette_curve(k_range, sils, name: str) -> str:
    plt.figure(figsize=(7, 5))
    plt.plot(list(k_range), sils, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title(f"Curva de Silhouette – {name}")
    plt.tight_layout()
    path = FIG_OUTDIR / f"kmeans_silhouette_curve_{name}_extra.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path)


def save_cluster_sizes(labels: np.ndarray, name: str) -> str:
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(7, 5))
    plt.bar(unique.astype(str), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Tamaño")
    plt.title(f"Tamaño de clusters – {name}")
    plt.tight_layout()
    path = FIG_OUTDIR / f"cluster_sizes_{name}_extra.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path)


def save_corr_heatmap(X: pd.DataFrame, name: str) -> str:
    corr = np.corrcoef(X.T)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"Matriz de correlación (características) – {name}")
    plt.xticks(ticks=np.arange(len(X.columns)), labels=np.arange(len(X.columns)), rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(len(X.columns)), labels=np.arange(len(X.columns)), fontsize=6)
    plt.tight_layout()
    path = FIG_OUTDIR / f"corr_heatmap_{name}_extra.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return str(path)


# ------------------------------
# Pipeline de proceso por dataset
# ------------------------------
def process_dataset_generic(df: pd.DataFrame, dataset_name: str) -> ClusteringResult:
    X = preprocess_dataframe(df)
    k_range = DEFAULT_K_RANGE
    best_k, sils, models = best_k_by_silhouette(X, k_range)
    km, labels, _ = models[best_k]
    pca_model, comp2d = pca_2d(X)

    figures = {}
    figures["scree"] = save_scree_plot(pca_model, dataset_name)
    figures["pca2d"] = save_pca2d_clusters(comp2d, labels, dataset_name)
    figures["silhouette_curve"] = save_silhouette_curve(k_range, sils, dataset_name)
    figures["cluster_sizes"] = save_cluster_sizes(labels, dataset_name)
    figures["corr_heatmap"] = save_corr_heatmap(X, dataset_name)

    return ClusteringResult(
        name=dataset_name,
        n_samples=len(X),
        n_features=X.shape[1],
        best_k=best_k,
        labels=labels,
        pca_components_2d=comp2d,
        silhouette_scores=sils,
        k_range=list(k_range),
        figure_paths=figures
    )


# ------------------------------
# Entrypoint de ejemplo
# ------------------------------
def run_extra_analytics(
    bank_csv_path: str = "BankChurners.csv",
    hotel_csv_path: str = "hotel_bookings_muestra.csv",
    save_summary_csv: str = "resumen_kmeans_extra.csv",
):
    """Ejecuta el pipeline de análisis para banca y hotel usando rutas relativas
    al directorio de trabajo. Puedes ajustar las rutas si usas otra estructura.
    """
    # Carga
    df_bank = pd.read_csv(bank_csv_path)
    df_hotel = pd.read_csv(hotel_csv_path)

    # Proceso
    res_bank = process_dataset_generic(df_bank, "Banca")
    res_hotel = process_dataset_generic(df_hotel, "Hotel")

    # Resumen
    summary = pd.DataFrame([
        {"dataset": res_bank.name, "n_samples": res_bank.n_samples, "n_features": res_bank.n_features, "best_k": res_bank.best_k},
        {"dataset": res_hotel.name, "n_samples": res_hotel.n_samples, "n_features": res_hotel.n_features, "best_k": res_hotel.best_k},
    ])
    summary.to_csv(save_summary_csv, index=False)

    print("=== Resumen de clusters ===")
    print(summary)
    print(f"\nFiguras guardadas en: {FIG_OUTDIR.resolve()}")
    print("Banca:", res_bank.figure_paths)
    print("Hotel:", res_hotel.figure_paths)


# Si quieres que corra cuando se invoque directamente este archivo,
# descomenta la siguiente sección main.
if __name__ == "__main__":
    try:
        # Ajusta las rutas si ejecutas desde otra carpeta.
        run_extra_analytics(
            bank_csv_path="BankChurners.csv",
            hotel_csv_path="hotel_bookings_muestra.csv",
            save_summary_csv="resumen_kmeans_extra.csv",
        )
    except Exception as e:
        print("[ADVERTENCIA] No se pudo ejecutar run_extra_analytics automáticamente:", str(e))
        print("Sugerencia: verifica las rutas a los CSV o ejecuta run_extra_analytics() manualmente.")
# =========================== FIN BLOQUES AÑADIDOS ============================
