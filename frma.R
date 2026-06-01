# Clear objects from memory and run garbage collection
rm(list = ls())
gc()

# Load necessary libraries
library(affy)
library(frma)

# IMPORTANT: Ensure 'hgu133plus2frmavecs' is installed.
# You only need to run this command ONCE in your R console, not in the script itself.
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# BiocManager::install("hgu133plus2frmavecs")

# Define paths
# This should be the path to the directory that CONTAINS your GSE folders (e.g., /home/gpuhead-1/Downloads/rakhat/cel/cel/)
cel_files_path <- "/home/gpuhead-1/Downloads/rakhat/cel/cel/"
error_files_path <- "corrupt_CEL_files/" # Folder for problematic CEL files
output_path <- "normalized_output/" # Folder to save normalized CSVs

# Create directories if they don't exist
if (!dir.exists(error_files_path)) {
  dir.create(error_files_path)
}
if (!dir.exists(output_path)) {
  dir.create(output_path)
}

# Get all immediate subfolders within cel_files_path
# This will return paths like "/home/gpuhead-1/Downloads/rakhat/cel/cel/GSE11145", etc.
subfolders <- list.dirs(path = cel_files_path, full.names = TRUE, recursive = FALSE)

if (length(subfolders) == 0) {
  stop("No subfolders found in the specified CEL files path. Please check your directory structure.")
}

# Loop through each subfolder found
for (folder in subfolders) {
  # Extract the name of the current subfolder (e.g., "GSE11145")
  folder_name <- basename(folder)
  message(paste("\n--- Processing folder:", folder_name, "---"))
  
  # Get all CEL files within the current subfolder
  # recursive = FALSE ensures we only look directly inside 'folder'
  all_cel_files_in_folder <- list.celfiles(path = folder, full.names = TRUE, recursive = FALSE)
  
  if (length(all_cel_files_in_folder) == 0) {
    message(paste("No CEL files found in folder:", folder_name, ". Skipping to next folder."))
    next # Skip to the next folder in the loop
  }
  
  # --- Step 1: Identify Expected Dimension for CEL files in the current folder ---
  cel_dimensions <- list()
  for (file in all_cel_files_in_folder) {
    tryCatch({
      raw_data <- ReadAffy(filenames = file)
      expr_matrix <- exprs(raw_data)
      cel_dimensions[[file]] <- dim(expr_matrix)
    }, error = function(e) {
      message(paste("Skipping file due to error (initial dimension check):", basename(file), " - ", e$message))
    })
  }
  
  expected_rows <- NULL
  if (length(cel_dimensions) > 0) {
    # Find the most common row count among CEL files in this specific folder
    unique_dimensions_rows <- table(sapply(cel_dimensions, function(d) d[1]))
    expected_rows <- as.numeric(names(unique_dimensions_rows)[which.max(unique_dimensions_rows)])
    message(paste("Most common row count found for", folder_name, ":", expected_rows))
  } else {
    message(paste("No valid CEL files found to determine dimensions in folder:", folder_name, ". Skipping normalization for this folder."))
    next # Skip to the next folder
  }
  
  # --- Step 2: Process Only Valid CEL Files (matching expected dimensions) in the current folder ---
  valid_cel_files_in_folder <- c()
  for (file in all_cel_files_in_folder) {
    tryCatch({
      raw_data <- ReadAffy(filenames = file)
      expr_matrix <- exprs(raw_data)
      
      if (nrow(expr_matrix) == expected_rows) {
        valid_cel_files_in_folder <- c(valid_cel_files_in_folder, file)
      } else {
        message(paste("Moving file with incorrect dimensions from", folder_name, ":", basename(file)))
        file.rename(file, file.path(error_files_path, basename(file)))
      }
    }, error = function(e) {
      message(paste("Moving corrupt file from", folder_name, ":", basename(file), " - ", e$message))
      file.rename(file, file.path(error_files_path, basename(file)))
    })
  }
  
  # --- Step 3: Normalize the Valid CEL Files in the current folder and save ---
  if (length(valid_cel_files_in_folder) > 0) {
    normalized_frma_data <- NULL # Initialize to NULL for robust error checking
    
    tryCatch({
      # Read CEL files into an AffyBatch object for this specific folder
      valid_frma_data <- ReadAffy(filenames = valid_cel_files_in_folder)
      
      # Perform FRMA normalization on this folder's data
      normalized_frma_data <- frma(valid_frma_data)
      
      # Crucial check: Ensure frma returned a valid ExpressionSet object
      if (is.null(normalized_frma_data) || !is(normalized_frma_data, "ExpressionSet")) {
        stop("FRMA normalization did not return a valid ExpressionSet object. This might indicate an issue with the input data, platform, or required data packages (e.g., 'hgu133plus2frmavecs' for U133 Plus 2.0 arrays).")
      }
      
      # Extract the expression matrix
      normalized_frma_expr <- as.data.frame(exprs(normalized_frma_data))
      
      # Define the output filename based on the current folder's name (e.g., "GSE11145_normalized_expression.csv")
      output_file_name <- paste0(folder_name, "_normalized_expression.csv")
      output_file_path <- file.path(output_path, output_file_name)
      
      # Save the normalized data to a CSV file
      write.csv(normalized_frma_expr, file = output_file_path, row.names = TRUE)
      message(paste("Normalization successful and saved for", folder_name, "as", output_file_name))
      
    }, error = function(e) {
      # Catch and report errors specifically during the normalization step for this folder
      message(paste("ERROR: FRMA normalization failed for folder", folder_name, ". Reason:", e$message))
      message(paste("Skipping saving normalized data for folder:", folder_name))
    })
  } else {
    message(paste("No valid CEL files found for normalization in folder:", folder_name, ". Skipping normalization for this folder."))
  }
}

message("\n--- Script finished processing all folders ---")