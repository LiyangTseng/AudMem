<?php

    // sql database info 
    $servername = "ns29.g-dns.com";
    $username = "musicme1_admin";
    $password = "5MLab3zQI(N*";
    $dbname = "musicme1_database_01";

    $email = $_POST['email'];

    $TestLink = mysql_connect($servername, $username, $password);
    $db_selected = mysql_select_db($dbname, $TestLink);
    $sql = "SELECT * FROM experimentData where userEmail='$email'";

    $Result = mysql_query($sql, $TestLink);
    if (mysql_num_rows($Result) == 0) {
        echo "Database Error";
        exit;
    }
    
    // echo mysql_num_rows($Result) . " results";

    mysql_close($TestLink);
    
?>
