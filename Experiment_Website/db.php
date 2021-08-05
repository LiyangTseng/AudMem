<?php
    function console_log( $data ){
        echo '<script>';
        echo 'console.log('. json_encode( $data ) .')';
        echo '</script>';
    }
    function select() {
        echo "The select function is called.";
        exit;
    }
    
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


    if (isset($_POST['action'])) {
        console_log($_POST['action']);
        switch ($_POST['action']) {
            case 'heard':
                echo "The heard function is called.";
                $sql = "UPDATE test SET lastname='Ron' WHERE id=2";
                if ($conn->query($sql) === TRUE) {
                    echo "Record updated successfully";
                } else {
                    echo "Error updating record: " . $conn->error;
                }
                break;
            case 'select':
                select();
                break;
        }
    }



    $conn->close();
?>