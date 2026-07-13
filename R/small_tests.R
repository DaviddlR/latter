


test_scarf_iris = function() {
  message("Training scarf on iris")


  # 1. Definir la receta
  # Usamos Species como Target. Las numéricas serán procesadas por SCARF.
  rec_spec_iris <- recipes::recipe(Species ~ ., data = iris) |>
    step_extract_latent(recipes::all_numeric_predictors(), epochs = 2)

  # 2. Entrenar la receta (Ejecuta el PREP y entrena SCARF)
  message("Entrenando SCARF con iris...")
  rec_trained_iris <- recipes::prep(rec_spec_iris, training = iris)

  # 3. Aplicar la receta (Ejecuta el BAKE)
  message("Aplicando transformación latente...")
  iris_transformed <- recipes::bake(rec_trained_iris, new_data = iris)

  # 4. Verificaciones de éxito:
  print(dim(iris_transformed))
  # Debería mostrar: [150 filas, 257 columnas]
  # (1 de la variable Species + 256 de las dimensiones latentes)

  dplyr::glimpse(iris_transformed[, 1:5])


}




test_scarf_unswnb15 = function() {

  # 1. Definir la receta con el orden correcto de preprocesamiento
  rec_spec_unsw <- recipes::recipe(attack_cat ~ ., data = df_train) |>
    recipes::step_rm(label) |>  # Exclude columns
    recipes::step_novel(recipes::all_nominal_predictors(), new_level = "unknown") |>
    recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE) |>
    recipes::step_zv(recipes::all_predictors()) |>
    recipes::step_normalize(recipes::all_numeric_predictors()) |>
    # Al usar all_numeric_predictors(), ya NO incluirá ni 'attack_cat' ni 'label'
    step_extract_latent(recipes::all_numeric_predictors(), epochs = 2, batch_size = 512)

  # 2. Entrenar la receta (PREP)
  # Esto dummificará, normalizará y entrenará tu modelo SCARF
  message("Entrenando pipeline completo con UNSW-NB15...")
  rec_trained_unsw <- recipes::prep(rec_spec_unsw, training = df_train)

  # 3. Procesar los datos (BAKE)
  message("Transformando el dataset UNSW-NB15...")
  unsw_transformed <- recipes::bake(rec_trained_unsw, new_data = df_train)

  # 4. Verificaciones de éxito:
  print(dim(unsw_transformed))
  # Debería tener las mismas filas que tu dataset original,
  # y exactamente 257 columnas (attack_cat + 256 de SCARF)

  # Comprobar que attack_cat sigue ahí y las variables viejas desaparecieron
  sum(c("label", "dur", "proto", "service") %in% names(unsw_transformed))
  # Debería dar 0 (las variables originales numéricas y dummificadas fueron reemplazadas)

  "attack_cat" %in% names(unsw_transformed)
  # Debería dar TRUE

  names(unsw_transformed)[!grepl("extracted_dim_", names(unsw_transformed))]
}
