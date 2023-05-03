<?php

require_once("init.php");

// nacitame aktualne data
$parametersJson = getData();

// pridame textovu spravu (hint) o aktualnom stave ESP
if (array_key_exists(PARAM_HINT, $parametersJson)) {
    $parametersJson['hintText'] = $hints[$parametersJson['hint']];
}

// nastavime header response na json
header('Content-Type: application/json; charset=utf-8');

// vratime textovu podobu json
echo json_encode($parametersJson);
exit();
