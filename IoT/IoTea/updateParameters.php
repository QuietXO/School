<?php

// nacitame globalne nastavenia
require_once("init.php");

// funkcia zisti, ci ide o prve vlozenie dat po (re)starte ESP
// ak mame vsetky parametre, ide o prve/vstupne data
function isInitialUpdate(array $keys)
{
    foreach ($keys as $key) {
        if (!isset($_GET[$key])) {
            return false;
        }
    }
    return true;
}

// nacitame data zo suboru alebo vytvorime nove pole
if (file_exists($filename) && !isInitialUpdate($keys)) {
    $parametersJson = getData();
    $parametersJson[PARAM_UPDATED_AT] = $datetime;
} else {
    $parametersJson = [
        PARAM_CREATED_AT => $datetime,
        PARAM_UPDATED_AT => null,
    ];
}

// updatneme parametre
foreach ($keys as $key) {
    if (isset($_GET[$key])) {
        $parametersJson[$key] = $_GET[$key];
    }
}

// ulozime updatnute parametre do suboru
setData($parametersJson);