<?php

    // sql database info 
    $servername = "ns29.g-dns.com";
    $username = "musicme1_admin";
    $password = "5MLab3zQI(N*";
    $dbname = "musicme1_database_01";

    $startTime = $_POST['startTime'];
    $email = $_POST['email'];
    $audioOrderStr = $_POST['audioOrderStr'];
    $responseStr = $_POST['responseStr'];
    $responsePositionStr = $_POST['responsePositionStr'];
    $experimentFinished = $_POST['experimentFinished'];

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    if ($email == "DB_Error") {
        echo "Database Error";
        exit;
    }
    $sql = "INSERT INTO experimentData (startTime, updateTime, userEmail, audioOrder, userResponse, responsePosition, experimentFinished) VALUES ('$startTime', '$startTime', '$email', '$audioOrderStr', '$responseStr', '$responsePositionStr', '$experimentFinished')";

    if ($conn->query($sql) === TRUE) {
        echo "Record inserted successfully in insert_row_to_db.php";
    } else {
        echo "Database Error";
        exit;
        // pass to js to redirect
    }


    $conn->close();
    
?>
