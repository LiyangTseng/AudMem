<?php
    function console_log( $data ){
        // printing at console in .php file
            echo '<script>';
            echo 'console.log('. json_encode( $data ) .')';
            echo '</script>';
    }
    

    $EmailSubmitted=$_POST['order'];
    echo '接收到的內容為: '.$EmailSubmitted;
    console_log($EmailSubmitted);


    // sql database info 
    $servername = "sql202.epizy.com";
    $username = "epiz_29320404";
    $password = "CrNYzDv8IuI7";
    $dbname = "epiz_29320404_MusicMem";

    // Create connection
    $conn = new mysqli($servername, $username, $password, $dbname);
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }

    // update order
    foreach ($_POST as $key => $value) {
        # code...
        $sql = "UPDATE INTO experimentOrderBeta (time, email) VALUES (CURRENT_TIMESTAMP, '$EmailSubmitted')";


        if ($conn->query($sql) === TRUE) {
            echo "Record updated successfully";
        } else {
            echo "Error updating record: " . $conn->error;
        }
    
    }
            
    $conn->close();

    // redirect to the experiment page
    header("Location: experiment.html");
    die();

?>
