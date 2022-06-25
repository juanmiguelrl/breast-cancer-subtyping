if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("TCGAbiolinks")
require(TCGAbiolinks)

args <- commandArgs(trailingOnly = TRUE)

brca_subtypes <- TCGAbiolinks::TCGAquery_subtype("brca")
output_path =  (args[1])
write.csv(brca_subtypes,output_path)