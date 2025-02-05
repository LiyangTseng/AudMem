<?php

if (isset($_GET['msg'])){ // when ?msg=error

  $msg = $_GET['msg'];
  ?>

  <!--
    display your error
  -->

<!DOCTYPE html>
<html>
    <script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
    <head>
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KyZXEAg3QhqLMpG8r+8fhAXLRk2vvoC2f3B09zVXn8CA5QIVfZOJ3BCsw2P0p/We" crossorigin="anonymous">
        <title> Error Page</title>
    </head>

    <!-- <h1><?php echo $msg; ?></h1> -->
    <body>
        <main>
            <div class="main_block position-relative overflow-hidden p-3 p-md-5 text-center ">
                <div class=" col-12 col-lg-8 p-lg-4 mx-auto my-3">
                  <h1 class="title display-2 font-weight-normal">Error querying database !!<br> Please contact recruiter</h1>
                </div>
            </div>            
        </main>
        <footer id=scroll-target>
            <div class="container">
                <!--Grid row-->
                <div class="row py-4">
                  <!--Grid column-->
                  <div class="col-md-12 text-center">
                    <i class="material-icons" style="font-size:56px">email</i>

                    <p style="font-size:36px">liyangtseng.ee06@nycu.edu.tw</p>
                  </div>
                  <!--Grid column-->
                </div>
                <!--Grid row-->
              </div>
        </footer>
        
    
    </body>
</html>


<?php }else{ ?>

<!--
display home page
-->
<!DOCTYPE html>
<html>
    <script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
    <head>
        <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KyZXEAg3QhqLMpG8r+8fhAXLRk2vvoC2f3B09zVXn8CA5QIVfZOJ3BCsw2P0p/We" crossorigin="anonymous">
        <link rel="stylesheet" href="home.css">
        <title> Experiment on Audio Memorability</title>
    </head>
    <body>
        <header>
            <nav class="navbar navbar-expand-lg py-0 navbar-dark">
                <div class="container-fluid">
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNavAltMarkup" aria-controls="navbarNavAltMarkup" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                      </button>
                      <div class="collapse navbar-collapse" id="navbarNavAltMarkup">
                        <div class="navbar-nav">
                          <a class="nav-link active" aria-current="page" href="index.html">Home</a>
                          <a class="nav-link" href="#scroll-target">Contact</a>
                        </div>
                      </div>
                </div>
            </nav>
        </header>
        <main>
            <div class="main_block position-relative overflow-hidden p-3 p-md-5 text-center ">
                <div class=" col-12 col-lg-8 p-lg-4 mx-auto my-3">
                  <h1 class="title display-2 font-weight-normal">Experiment on Audio Memorability</h1>
                  <a class="btn btn-dark btn-lg mt-5" href="experiment.html">START</a>
                </div>
            </div>            
        </main>
        <footer id=scroll-target>
            <div class="container">
                <!--Grid row-->
                <div class="row py-4">
                  <!--Grid column-->
                  <div class="col-md-12 text-center">
                    <i class="material-icons" style="font-size:36px">email</i>

                    <p>liyangtseng.ee06@nycu.edu.tw</p>
                  </div>
                  <!--Grid column-->
                </div>
                <!--Grid row-->
              </div>
        </footer>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.9.3/dist/umd/popper.min.js" 
            integrity="sha384-eMNCOe7tC1doHpGoWe/6oMVemdAVTMs2xqW4mwXrXsW0L84Iytr2wi5v2QjrP/xp" 
            crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.0/dist/js/bootstrap.min.js" 
            integrity="sha384-cn7l7gDp0eyniUwwAZgrzD06kc/tftFf19TOAs2zVinnD/C7E91j9yyk5//jjpt/" 
            crossorigin="anonymous"></script>
    
    </body>
</html>



<?php }

?>
