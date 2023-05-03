<?php
        echo '<h1>Web page with parameters</h1>';

        echo '<h2><a href="https://eheeey.azurewebsites.net">
      	Home Page</a></h2>';

        echo '<p> https://eheeey.azurewebsites.net/index2.php?A=10&B=2 </p>';

        $sn1 = $_GET["A"];
        $sn2 = $_GET["B"];

        $text = "A = " . $sn1 . " B = " . $sn2;
        $sum = $sn1 + $sn2;

        echo $text;
        echo "<br>";
        echo "Sum: " . $sum;
?>
