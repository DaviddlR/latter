# This is the latter README. Please, read me carefully


En este punto, el bloque sin recipes funciiona bien y hace la corrupción teniendo en cuenta los grupos de columnas generados por
One hot encoding. La parte de recipes es la que no tiene implementada esto.

Si en un futuro se lleva a cabo la idea de forzar al usuario a que use step_dummy(one_hot = TRUE) sin cambiar nombres de columnas,
se puede reutilizar la función que tengo en utils para detectar grupos con expresiones regulares y hacer un flujo bastante similar
al que se usa en el procesamiento sin recipes. 

En read_data > prepare_scarf_data hay un comentario con algo más de info. Good luck.
