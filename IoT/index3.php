<?php
        echo '<h1>Create .txt based on input parameters</h1>';

        echo '<h2><a href="https://eheeey.azurewebsites.net">
        Home Page</a></h2>';

        echo '<p> https://eheeey.azurewebsites.net/index3.php?A=10&B=2 </p>';

        $sn1 = $_GET["A"];
        $sn2 = $_GET["B"];

        $file1 = fopen("sensors.txt","w") or die("Unable to open file!");
        $text1 = "A = " . $sn1 . " B = " . $sn2;

        fwrite($file1, $text1);
        fclose($file1);

        $file2 = fopen("actuator.txt","w") or die("Unable to open file!");
        $text2 = "Value from actuator. Save this value to actuator.txt";
        fwrite($file2, $text2);
        fclose($file2);

        $file3 = fopen("actuator.txt","r") or die ("Subor neexistuje");
        $text3 = fread($file3,filesize("actuator.txt"));
        echo $text3;
        fclose($file3);
?>
