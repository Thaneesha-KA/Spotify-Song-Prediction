# ---------------------------------------------------
# 1. Load Libraries
# ---------------------------------------------------
library(tidyverse)
library(data.table)
library(cluster)
library(factoextra)
library(caret)
library(zoo)

# ---------------------------------------------------
# 2. Load Dataset
# ---------------------------------------------------
file_path <- "C:/Users/Thaneesha/Downloads/spotify-2023.csv"
raw <- fread(file_path, stringsAsFactors = FALSE, data.table = FALSE)

cat("✅ Dataset loaded with", nrow(raw), "rows and", ncol(raw), "columns.\n")

# ---------------------------------------------------
# 3. Preprocessing (Numeric features only)
# ---------------------------------------------------
df <- raw %>% select(where(is.numeric))

if (ncol(df) == 0) stop("❌ No numeric columns detected!")

cat("✅ Found", ncol(df), "numeric features.\n")

# Handle missing values with median imputation
df_imputed <- as.data.frame(lapply(df, function(x) {
  if (is.numeric(x)) na.aggregate(x, FUN = median) else x
}))

# Scale (keep all features as-is, no PCA, no log transform)
df_scaled <- scale(df_imputed)

n_points <- nrow(df_scaled)
cat("✅ After preprocessing:", n_points, "rows,", ncol(df_scaled), "features.\n")

if (n_points < 2) stop("❌ Not enough data points for clustering.")

# ---------------------------------------------------
# 4. Find Optimal K (Elbow + Silhouette for K-Means)
# ---------------------------------------------------
max_k <- min(10, n_points - 1)
if (max_k < 2) stop("❌ Not enough points to test multiple cluster sizes.")

# Elbow Method
set.seed(123)
fviz_nbclust(df_scaled, kmeans, method = "wss") +
  labs(subtitle = "Elbow Method for Optimal K (K-Means)")

# Silhouette Method
set.seed(123)
sil_plot <- fviz_nbclust(df_scaled, kmeans, method = "silhouette") +
  labs(subtitle = "Silhouette Method for Optimal K (K-Means)")
print(sil_plot)

# Best K from silhouette
sil_width <- numeric(max_k - 1)
for (k in 2:max_k) {
  km <- kmeans(df_scaled, centers = k, nstart = 25)
  ss <- silhouette(km$cluster, dist(df_scaled))
  sil_width[k - 1] <- mean(ss[, 3])
}
best_sil <- which.max(sil_width) + 1
K <- best_sil
cat("✅ Best K (Silhouette):", K, "\n")

# ---------------------------------------------------
# 5. K-Means Clustering
# ---------------------------------------------------
set.seed(123)
km <- kmeans(df_scaled, centers = K, nstart = 50)
raw$KMeans_Cluster <- factor(km$cluster)

fviz_cluster(km, data = df_scaled, geom = "point", ellipse.type = "norm",
             main = paste("K-Means Clustering (K =", K, ")"))

# ---------------------------------------------------
# 6. Hierarchical Clustering
# ---------------------------------------------------
d <- dist(df_scaled)
hc <- hclust(d, method = "ward.D2")

fviz_dend(hc, k = K, cex = 0.5,
          rect = TRUE, rect_border = "jco", rect_fill = TRUE,
          main = paste("Hierarchical Clustering (K =", K, ")"))

raw$Hier_Cluster <- cutree(hc, k = K)

# ---------------------------------------------------
# 7. Save Results
# ---------------------------------------------------
output_path <- "C:/Users/Thaneesha/Downloads/spotify_clusters.csv"
write.csv(raw, output_path, row.names = FALSE)
cat("✅ Clustering results saved to:", output_path, "\n")


