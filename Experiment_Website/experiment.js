let vigilance_threshold = 0.3; // TODO: need to change after pilot study
let now_playing = document.querySelector(".now-playing");
let track_name = document.querySelector(".track-name");

let playpause_btn = document.querySelector(".playpause-track");
let next_btn = document.querySelector(".next-track");
let prev_btn = document.querySelector(".prev-track");

let seek_slider = document.querySelector(".seek_slider");
let volume_slider = document.querySelector(".volume_slider");
let curr_time = document.querySelector(".current-time");
let total_duration = document.querySelector(".total-duration");

let email = document.getElementById("userEmail").value;

let alhrd_checked = document.getElementById("toggle-alhrd").checked;
let unhrd_checked = document.getElementById("toggle-unhrd").checked;
let hrd_checked = document.getElementById("toggle-hrd").checked;

let track_index = -1;
let slot_cnt = 0;
let isPlaying = false;
let updateTimer;
let startTime = '';
let vigilanceDetail = 'N/A';
let vigilanceScore = -1;
let finished_token = 'N/A';
let experimentFinished = false;
let waitForLoad = false;
let waitTime = -1;
let timeoutSkip;
let timerStart;

// Create new audio element
let curr_track = document.createElement('audio');

let dir = 'clips/'
// Define the tracks that will be played, 
// notice the filename should contain only alphabet and number to prevent audio.src not found
let track_list = ['normalize_5s_intro_thc1MtNagC8.wav', 'normalize_5s_intro_Wo2qUD1g7xM.wav', 'normalize_5s_intro_3ObVN3QQiZ8.wav', 'normalize_5s_intro_S-zQJFRX5Fg.wav', 'normalize_5s_intro_SyZOAgXiPMw.wav', 'normalize_5s_intro_GQT8ejgV2_A.wav', 'normalize_5s_intro_PQAIxeSIQU4.wav', 'normalize_5s_intro_E-8pyVBvCPQ.wav', 'normalize_5s_intro_Qr8eZSVaw10.wav', 'normalize_5s_intro_p7j-tz1Cn4o.wav', 'normalize_5s_intro_nISI4qF55F4.wav', 'normalize_5s_intro_RoeRU5zxkak.wav', 'normalize_5s_intro_EygNk739nnY.wav', 'normalize_5s_intro_w1G3rqVil1s.wav', 'normalize_5s_intro_KKc_RMln5UY.wav', 'normalize_5s_intro_Ng2JdroNfC0.wav', 'normalize_5s_intro_xc0sWhVhmkw.wav', 'normalize_5s_intro_VVRszjvg3_U.wav', 'normalize_5s_intro_C7u6rtswjCU.wav', 'normalize_5s_intro_HiPkwl5p1GY.wav', 'normalize_5s_intro_mYa_9d2Daas.wav', 'normalize_5s_intro_6MSYrN4YfKY.wav', 'normalize_5s_intro_O2q_9lBDM7I.wav', 'normalize_5s_intro_7E_a_VKjcl8.wav', 'normalize_5s_intro_a8cJLohQ_Jg.wav', 'normalize_5s_intro_7zz-nEVKZdc.wav', 'normalize_5s_intro_JeGhUESd_1o.wav', 'normalize_5s_intro_IN1f9k8qVDk.wav', 'normalize_5s_intro_RhBb77hG0iw.wav', 'normalize_5s_intro_qAiwzv8N7rM.wav', 'normalize_5s_intro_AoB8koE95C0.wav', 'normalize_5s_intro_j3DigipQ_hQ.wav', 'normalize_5s_intro_1X0SdKtnwo8.wav', 'normalize_5s_intro_RCJx5VW-fQI.wav', 'normalize_5s_intro_S_-qkv0NZ1g.wav', 'normalize_5s_intro_C90sY_Ht6Ig.wav', 'normalize_5s_intro_Z5gvqq3ChII.wav', 'normalize_5s_intro_zumMQrI_tMg.wav', 'normalize_5s_intro_gwsaElRJI2M.wav', 'normalize_5s_intro_ftjEcrrf7r0.wav', 'normalize_5s_intro_ZBBS4imv1qo.wav', 'normalize_5s_intro_DyQ_9p6y89c.wav', 'normalize_5s_intro_vgZv7Uu4YrA.wav', 'normalize_5s_intro_wcLXjQLwSBE.wav', 'normalize_5s_intro_7LuQQP-DAoc.wav', 'normalize_5s_intro_BEo0rqOZIng.wav', 'normalize_5s_intro_n4HTXYR-2AI.wav', 'normalize_5s_intro_72T4j04MS8o.wav', 'normalize_5s_intro_6TT_UgrRHq8.wav', 'normalize_5s_intro_uo8qDCDZhK0.wav', 'normalize_5s_intro_Et-YdmSo_3A.wav', 'normalize_5s_intro_oxKbrl4kyg8.wav', 'normalize_5s_intro_XgwqnGG-pbI.wav', 'normalize_5s_intro_1wpJkzCWHcI.wav', 'normalize_5s_intro_bwQ49N0jVvE.wav', 'normalize_5s_intro_OMR2W-7AyYU.wav', 'normalize_5s_intro_sjlkxcwhpwA.wav', 'normalize_5s_intro_4F1wvsJXXVY.wav', 'normalize_5s_intro_YEq-cvq_cK4.wav', 'normalize_5s_intro_42O51bcJyq0.wav', 'normalize_5s_intro_5FYAICvv-d0.wav', 'normalize_5s_intro_yBzk2xXE9yg.wav', 'normalize_5s_intro_zEWSSod0zTY.wav', 'normalize_5s_intro_dvf--10EYXw.wav', 'normalize_5s_intro_xQOXxmznGPg.wav', 'normalize_5s_intro_hMWoOunsMFM.wav', 'normalize_5s_intro_TnsOVDCq_b0.wav', 'normalize_5s_intro_Yh78Ll6-ODQ.wav', 'normalize_5s_intro_IYnu4-69fTA.wav', 'normalize_5s_intro_SubIr_Fyp4M.wav', 'normalize_5s_intro_WrRAZVJGImw.wav', 'normalize_5s_intro_gFnNr5vr5bQ.wav', 'normalize_5s_intro_j9KKh215HTs.wav', 'normalize_5s_intro_XBTT9tSVsh0.wav', 'normalize_5s_intro_u8BxVzRG9bE.wav', 'normalize_5s_intro_SQBuVfTX1ME.wav', 'normalize_5s_intro_-MqZKMbOYEA.wav', 'normalize_5s_intro_IpniN1Wq68Y.wav', 'normalize_5s_intro_1lunUbvf35M.wav', 'normalize_5s_intro_zk04E79riMQ.wav', 'normalize_5s_intro_uUfPwlxFFJM.wav', 'normalize_5s_intro_Ws-QlpSltr8.wav', 'normalize_5s_intro_xT1eOeXlTXg.wav', 'normalize_5s_intro_1Ngn3fZIK2E.wav', 'normalize_5s_intro_2JL_KcEzkqg.wav', 'normalize_5s_intro_4jvQFLlRQlo.wav', 'normalize_5s_intro_AjGkbFqi67c.wav', 'normalize_5s_intro_ahpmuikko3U.wav', 'normalize_5s_intro_sY5wXfgspQI.wav', 'normalize_5s_intro_HMvXE4Zs6ZA.wav', 'normalize_5s_intro_gv7BRXvZJbI.wav', 'normalize_5s_intro_4wgo8K28RNM.wav', 'normalize_5s_intro_2ySLmwsfP4Q.wav', 'normalize_5s_intro_MY4YJxn-9Og.wav', 'normalize_5s_intro_3gjfHYZ873o.wav', 'normalize_5s_intro_csHiDQXIggE.wav', 'normalize_5s_intro_C5VCGM2J5ls.wav', 'normalize_5s_intro_ey4Fc9DP5Rw.wav', 'normalize_5s_intro_bI7xde9-3BI.wav', 'normalize_5s_intro_EfZ-dVDySzc.wav', 'normalize_5s_intro_Zh3uBgwow8A.wav', 'normalize_5s_intro_JQTlG7NxJek.wav', 'normalize_5s_intro_1CrxzClzLvs.wav', 'normalize_5s_intro_0aC-jOKuBFE.wav', 'normalize_5s_intro_xePw8n4xu8o.wav', 'normalize_5s_intro_lEHM9HZf0IA.wav', 'normalize_5s_intro_xhmtXrtLkgo.wav', 'normalize_5s_intro_hHItMz0gfaU.wav', 'normalize_5s_intro_99f0oH45TVc.wav', 'normalize_5s_intro_co6WMzDOh1o.wav', 'normalize_5s_intro_xqzOxMdhmzU.wav', 'normalize_5s_intro_h-nnAeByB1A.wav', 'normalize_5s_intro_TFv9Kcym9dg.wav', 'normalize_5s_intro_tEW2eRQ-4DY.wav', 'normalize_5s_intro_VAc0xuVa7jI.wav', 'normalize_5s_intro_PALMMqZLAQk.wav', 'normalize_5s_intro_STpRa2JPFA0.wav', 'normalize_5s_intro_SgJMnEdtTXA.wav', 'normalize_5s_intro_NL2ZHPji3Z0.wav', 'normalize_5s_intro_EVSuxb6Ywcg.wav', 'normalize_5s_intro_wAJMhJpSCIc.wav', 'normalize_5s_intro_GphIn74Weu0.wav', 'normalize_5s_intro_gue_crpFdSE.wav', 'normalize_5s_intro_oQ0O2cd1T04.wav', 'normalize_5s_intro_vMcFA2x23FE.wav', 'normalize_5s_intro_FhvXg70ycrM.wav', 'normalize_5s_intro_lE_747E_Sdg.wav', 'normalize_5s_intro_i0MrGb1hT2U.wav', 'normalize_5s_intro_bI8-2blisUM.wav', 'normalize_5s_intro_aQ06TfyA1Ks.wav', 'normalize_5s_intro_ZvrysfBDzSs.wav', 'normalize_5s_intro_v2seHL0pwbg.wav', 'normalize_5s_intro_BrrWNfjgHGs.wav', 'normalize_5s_intro_j1c70vRHdhQ.wav', 'normalize_5s_intro_3DCHLwOqtJs.wav', 'normalize_5s_intro_g20t_K9dlhU.wav', 'normalize_5s_intro_EH1OEWJ9C5w.wav', 'normalize_5s_intro_SCBxmcwmX7U.wav', 'normalize_5s_intro_tXvpe2GbUec.wav', 'normalize_5s_intro_7ZgPGMfUVek.wav', 'normalize_5s_intro_aIJuCcGFJkc.wav', 'normalize_5s_intro_RLMl1umHgp0.wav', 'normalize_5s_intro_KT-m6qTJyN0.wav', 'normalize_5s_intro_WJs-_T8I74Y.wav', 'normalize_5s_intro_aIyqRdrHodE.wav', 'normalize_5s_intro_XJT-fM4nBJU.wav', 'normalize_5s_intro_7QQzDQceGgU.wav', 'normalize_5s_intro_fE2h3lGlOsk.wav', 'normalize_5s_intro_Oq1n8fUxQZc.wav', 'normalize_5s_intro_pssWSj42t8M.wav', 'normalize_5s_intro_GsPq9mzFNGY.wav', 'normalize_5s_intro_Jg9NbDizoPM.wav', 'normalize_5s_intro_Ib7m3Qh-4O4.wav', 'normalize_5s_intro_hn3wJ1_1Zsg.wav', 'normalize_5s_intro_hjZqVw3qI9E.wav', 'normalize_5s_intro_cUKD9tEeBp0.wav', 'normalize_5s_intro_q_4no3KCrY4.wav', 'normalize_5s_intro_VlWs8ey2nyg.wav', 'normalize_5s_intro_Srp0opA8V8o.wav', 'normalize_5s_intro_PYM9NUU9Roc.wav', 'normalize_5s_intro_v0UvOsCi8mc.wav', 'normalize_5s_intro_zaCbuB3w0kg.wav', 'normalize_5s_intro_PCp2iXA1uLE.wav', 'normalize_5s_intro_S2RnxiNJg0M.wav', 'normalize_5s_intro_Jtv4satRsP0.wav', 'normalize_5s_intro_ytq5pGcM77w.wav', 'normalize_5s_intro_9nWpMZFrbvI.wav', 'normalize_5s_intro_1kN-34GFMYM.wav', 'normalize_5s_intro_Yyvo9O8fN-A.wav', 'normalize_5s_intro_ulj-L3K_Gzs.wav', 'normalize_5s_intro_V-ar6MLjy5o.wav', 'normalize_5s_intro_dtOv6WvJ44w.wav', 'normalize_5s_intro_XkC8Uzl9pCY.wav', 'normalize_5s_intro_jII5qoCrzYE.wav', 'normalize_5s_intro_7pcZIsJNlAs.wav', 'normalize_5s_intro_0QN9KLFWn7I.wav', 'normalize_5s_intro_d6BzCEkGd3I.wav', 'normalize_5s_intro_lYxcW8jtFw0.wav', 'normalize_5s_intro_R1T_SrdQGH8.wav', 'normalize_5s_intro_YOKq1VmEbtc.wav', 'normalize_5s_intro_19Q9l85Feqw.wav', 'normalize_5s_intro_CXm7hPs_als.wav', 'normalize_5s_intro_nFOLhtsyvMA.wav', 'normalize_5s_intro_-8cFfkyk7vA.wav', 'normalize_5s_intro_ZIiQ1jMqhVM.wav', 'normalize_5s_intro_hejXc_FSYb8.wav', 'normalize_5s_intro_eXvBjCO19QY.wav', 'normalize_5s_intro_haCay85cpvo.wav', 'normalize_5s_intro_RpJz01guPMY.wav', 'normalize_5s_intro_sPlXrbVLdO8.wav', 'normalize_5s_intro_Mme9REVuidw.wav', 'normalize_5s_intro_UGTYqTKUl8w.wav', 'normalize_5s_intro_9DP0yMwvyWE.wav', 'normalize_5s_intro_WrDJMxSKlCA.wav', 'normalize_5s_intro_2F8Kr91wQ0U.wav', 'normalize_5s_intro_gyegm85BPPA.wav', 'normalize_5s_intro_Xhh3_-JRnDc.wav', 'normalize_5s_intro_WRSeV_27z6k.wav', 'normalize_5s_intro_HwcCBnfhsR4.wav', 'normalize_5s_intro_bd5m12UEHWI.wav', 'normalize_5s_intro_1juIFmPyG-Y.wav', 'normalize_5s_intro_DGsoqhIUgDQ.wav', 'normalize_5s_intro_2UL-1MOlSPw.wav', 'normalize_5s_intro_2AWE9tqnDPw.wav', 'normalize_5s_intro_68b_HImZAig.wav', 'normalize_5s_intro_GIulOhzXufc.wav', 'normalize_5s_intro_Stet_4bnclk.wav', 'normalize_5s_intro_RHGfkuJv0j0.wav', 'normalize_5s_intro_0uLI6BnVh6w.wav', 'normalize_5s_intro_uo6VU4euIbY.wav', 'normalize_5s_intro_6pARjpdqxYQ.wav', 'normalize_5s_intro_hjIhCG_nIPA.wav', 'normalize_5s_intro_hV-FwW1LgxU.wav', 'normalize_5s_intro_mWfWyhzC22U.wav', 'normalize_5s_intro_IISA6t-9zzc.wav', 'normalize_5s_intro_gDevCxVY_wA.wav', 'normalize_5s_intro_IrtcCSE2bVY.wav', 'normalize_5s_intro_feVUoKhP1mE.wav', 'normalize_5s_intro_Tfypj4UwvvA.wav', 'normalize_5s_intro_TeH7sCVCMJk.wav', 'normalize_5s_intro_0EVVKs6DQLo.wav', 'normalize_5s_intro_d7to9URtLZ4.wav', 'normalize_5s_intro_TzhhbYS9EO4.wav', 'normalize_5s_intro_nn5nypm7GG8.wav', 'normalize_5s_intro_hed6HkYNA7g.wav', 'normalize_5s_intro_rWznOAwxM1g.wav', 'normalize_5s_intro_zyQkFh-E4Ak.wav', 'normalize_5s_intro_agKkcRXN2iE.wav', 'normalize_5s_intro_SZaZU_qi6Xc.wav', 'normalize_5s_intro_ZpDQJnI4OhU.wav', 'normalize_5s_intro_D4nWzd63jV4.wav', 'normalize_5s_intro_9odM1BRqop4.wav', 'normalize_5s_intro_F64yFFnZfkI.wav', 'normalize_5s_intro_Js2JQH_kt0I.wav', 'normalize_5s_intro_Skt_NKI4d6U.wav'];
// submit modal of user info
var submitModal = new bootstrap.Modal(document.getElementById("submitModal"), {
  keyboard: false,
  backdrop: 'static'
});
submitModal.show();
// bootstrap tooltip for decision hints
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl)
})

function generateToken(length) {
  var token           = '';
  var characters       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  var charactersLength = characters.length;
  for ( var i = 0; i < length; i++ ) {
    token += characters.charAt(Math.floor(Math.random() * 
charactersLength));
 }
 return token;
}

function shuffle(array) {
  // shuffle array order
  var currentIndex = array.length,  randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex], array[currentIndex]];
    }
    
  return array;
}

let audioOrder = [...Array(235).keys()];

// idx of audio
let vigilance_idx_arr = [0, 2, 15, 31, 54, 71, 84, 101, 110, 125, 137, 150, 157, 170, 178, 184, 188, 198, 216, 221, 232]

// order of every audio
let slotOrder = [0, 1, 2, 3, 4, 0, 5, 6, 2, 7, 8, 9, 1, 10, 11, 12, 13, 4, 14, 15, 16, 5, 17, 18, 19, 20, 
  15, 21, 7, 22, 23, 24, 25, 26, 27, 28, 29, 12, 30, 31, 32, 20, 33, 16, 34, 31, 14, 35, 36, 37, 27, 38, 39, 
  25, 40, 41, 42, 43, 44, 21, 45, 46, 32, 47, 48, 49, 30, 50, 51, 52, 53, 54, 35, 55, 56, 57, 58, 59, 60, 54, 
  61, 62, 63, 64, 65, 36, 66, 67, 68, 69, 70, 38, 71, 43, 72, 73, 74, 75, 40, 76, 71, 77, 53, 78, 72, 79, 51, 
  80, 81, 65, 82, 58, 47, 83, 84, 63, 85, 86, 87, 88, 84, 89, 90, 56, 91, 92, 93, 64, 94, 95, 68, 96, 97, 98, 
  74, 99, 6, 23, 100, 11, 101, 8, 102, 103, 104, 59, 101, 13, 22, 46, 100, 29, 105, 33, 48, 99, 106, 107, 45, 
  108, 109, 110, 105, 111, 61, 112, 102, 113, 114, 115, 110, 116, 117, 118, 119, 120, 121, 122, 86, 123, 124, 
  125, 108, 126, 127, 104, 128, 129, 125, 130, 121, 131, 132, 89, 107, 112, 93, 118, 133, 134, 75, 135, 136, 
  137, 138, 77, 139, 140, 82, 129, 141, 67, 142, 137, 135, 143, 144, 95, 145, 146, 147, 148, 149, 120, 150, 
  151, 152, 131, 153, 154, 134, 155, 150, 136, 156, 157, 158, 159, 160, 145, 138, 161, 141, 162, 157, 163, 
  164, 165, 166, 156, 167, 168, 169, 170, 160, 171, 164, 144, 172, 170, 153, 173, 161, 147, 174, 175, 151, 
  176, 149, 166, 116, 177, 178, 179, 113, 180, 17, 90, 178, 111, 181, 18, 127, 177, 76, 26, 128, 50, 182, 
  44, 142, 66, 179, 122, 41, 148, 183, 78, 83, 143, 184, 98, 185, 180, 162, 60, 184, 87, 186, 152, 187, 85, 
  188, 52, 165, 189, 182, 190, 188, 191, 154, 181, 192, 80, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 
  168, 203, 198, 204, 155, 205, 206, 193, 207, 208, 209, 210, 211, 212, 213, 196, 214, 215, 216, 62, 187, 158, 
  209, 217, 218, 216, 212, 190, 207, 219, 220, 221, 200, 204, 222, 223, 195, 221, 224, 225, 226, 227, 228, 
  229, 230, 214, 201, 215, 173, 211, 222, 171, 231, 224, 218, 232, 217, 174, 213, 223, 232, 219, 220, 233, 176, 225, 91, 226, 227, 234, 94];


let userResponses = [];
let userResponsePositions = [];
let userResponseInSec = 0;

function loadTrack(track_counter) {
  if (slotWithBreakOrder[track_counter] === -1) {
    // 3s break
    updateDB() // update userResponse of previous audio
    clearInterval(updateTimer);
    resetValues();
    curr_track.src = "pink_noise_3s.wav";

    // curr_track.load();
    
    track_name.textContent = `3 Seconds Break`;
    now_playing.textContent = "";
    $('label[for="toggle-hrd"]').hide();
    $('label[for="toggle-unhrd"]').hide();
    $('label[for="toggle-alhrd"]').hide();
    $('.skip_btn').hide();
    waitTime = 7000; // 7 second of waiting for loading
  } else if (slotWithBreakOrder[track_counter] === -2) {
    // 3m break
    updateDB(); // update userResponse of previous audio
    clearInterval(updateTimer);
    resetValues();
    curr_track.src = "pink_noise_3m.wav";
    // curr_track.load();
    
    track_name.textContent = `3 Minutes Break`;
    now_playing.textContent = "";
    $('label[for="toggle-hrd"]').hide();
    $('label[for="toggle-unhrd"]').hide();
    $('label[for="toggle-alhrd"]').hide();
    $('.skip_btn').hide();
    waitTime = 190000; // 3:10 of waiting for loading
  } else {
    // ordinay audio
    $('label[for="toggle-hrd"]').show();
    $('label[for="toggle-unhrd"]').show();
    $('label[for="toggle-alhrd"]').show();
    $('.skip_btn').hide();
    document.getElementById("toggle-alhrd").checked = false;
    document.getElementById("toggle-unhrd").checked = false;
    document.getElementById("toggle-hrd").checked = false;
    waitTime = 30000;
    track_index = audioOrder[slotWithBreakOrder[track_counter]];
    clearInterval(updateTimer);
    resetValues();
    curr_track.src =  dir + track_list[track_index];
    // curr_track.load();

    // track_name.textContent = `No.${track_index+1}`; // for debug prupose
    track_name.textContent = "";
    now_playing.textContent = "PLAYING " + (track_counter/2 + 1) + " OF " + slotOrder.length; 
  }
  curr_track.load();
  
  waitForLoad = true;
  timeoutSkip = setTimeout(() => {
    // wait for 30s, if audio still not loaded, pop up "skip" button
    if (waitForLoad == true) {
      $('.skip_btn').show();
    }
  }, waitTime);

  updateTimer = setInterval(seekUpdate, 1000);
  curr_track.addEventListener("ended", nextTrack);
}

function calculateVigilancePerformance() {
  let vigilance_performance = [];
  for (let index = 0; index < vigilance_idx_arr.length; index++) {
    const vigilance_idx = vigilance_idx_arr[index];
    first_occ = slotOrder.indexOf(vigilance_idx);
    last_occ = slotOrder.lastIndexOf(vigilance_idx);
    if (userResponses[first_occ] == 0 && userResponses[last_occ] == 1) {
      vigilance_performance.push(1); // memorized
    }
    else if (userResponses[first_occ] == 0 && userResponses[last_occ] == 0) {
      vigilance_performance.push(0); // not memorized
    }
    else {
      vigilance_performance.push(-1); // not count, ex already heard before
    }
  }
  const counts = {};

  for (const num of vigilance_performance) {
    counts[num] = counts[num] ? counts[num] + 1 : 1;
  }
  vigilanceScore = (counts[1] == 0) ? 0 : counts[1]/(counts[0]+counts[1]);
  vigilanceDetail = vigilance_performance.join();
}

function resetValues() {
  curr_time.textContent = "00:00";
  total_duration.textContent = "00:00";
  seek_slider.value = 0;
}

function playTrack() {
  curr_track.play();
  isPlaying = true;
}

function pauseTrack() {
  curr_track.pause();
  isPlaying = false;
  playpause_btn.innerHTML = '<i class="fa fa-play-circle fa-5x"></i>';;
}

function getUserResponse() {
  // wait until getting user responses
  return new Promise((resolve, reject) => {
    (function waitForResponse(){
      if (document.getElementById("toggle-alhrd").checked || document.getElementById("toggle-unhrd").checked || document.getElementById("toggle-hrd").checked) {
        $('label[for="toggle-alhrd"]').removeClass("glowed");
        $('label[for="toggle-unhrd"]').removeClass("glowed");
        $('label[for="toggle-hrd"]').removeClass("glowed");
        return resolve();
      }
      else {
        setTimeout(waitForResponse, 30);
        $('label[for="toggle-alhrd"]').addClass("glowed");
        $('label[for="toggle-unhrd"]').addClass("glowed");
        $('label[for="toggle-hrd"]').addClass("glowed");  
      }
    })();
  })  
}

function skip() {
  // if something went wrong when loading, user can press "skip" after 30 seconds of loading attempt
  if (slot_cnt%2 === 0){ 
    // if real audio(music) played in slot, record user responses
    userResponsePositions.push(-1); // -1 indicate skipping in userResponsePositions
    userResponseInSec = 0;
    userResponses.push(-2); // -2 indicate skipping in userResponses
  }
  // check experiment over or not
  if (slot_cnt < slotWithBreakOrder.length-1) {
    // experiment not over
    slot_cnt += 1;    
    loadTrack(slot_cnt);
    playTrack();  
  } else {
    // experiment over
    experimentFinished = true;
    finished_token = generateToken(16);
    calculateVigilancePerformance();
    if (vigilanceScore > vigilance_threshold) {
      $(".alert-success").show()
    }
    else {
      document.getElementById("failed_explain").textContent = "Your vigilance score " + vigilanceScore.toFixed(2) + " is lower than threshold " + vigilance_threshold; 
      $(".alert-danger").show()
    }
    
    updateDB();
    clearInterval(timerStart);

  }

}

async function nextTrack() {
  // disable skipping because music properly played
  clearTimeout(timeoutSkip);
  waitForLoad = false;

  if (slot_cnt%2 === 0){ 
    // if real audio(music) played in slot
    await getUserResponse();
    userResponsePositions.push(userResponseInSec);
    userResponseInSec = 0;
  
    if (document.getElementById("toggle-alhrd").checked) {
      // already heard: -1
      userResponses.push(-1);
    } else if (document.getElementById("toggle-unhrd").checked) {
      // unheard: 0
      userResponses.push(0);
    } else if (document.getElementById("toggle-hrd").checked) {
      // repeat: 1
      userResponses.push(1);
    }

  }
  // check experiment over or not
  if (slot_cnt < slotWithBreakOrder.length-1) {
    // experiment not over
    slot_cnt += 1;    
    loadTrack(slot_cnt);
    playTrack();  
  } else {
    // experiment over
    experimentFinished = true;
    finished_token = generateToken(16);
    calculateVigilancePerformance();
    if (vigilanceScore > vigilance_threshold) {
      $(".alert-success").show()
    }
    else {
      document.getElementById("failed_explain").textContent = "Your vigilance score " + vigilanceScore.toFixed(2) + " is lower than threshold " + vigilance_threshold; 
      $(".alert-danger").show()
    }
    
    updateDB();
    clearInterval(timerStart);
  }
}

function setVolume() {
  // TODO: make work instaneously (when mouse still pressing)
  curr_track.volume = volume_slider.value / 100;
}

function seekUpdate() {
  let seekPosition = 0;

  if (!isNaN(curr_track.duration)) {
    seekPosition = curr_track.currentTime * (100 / curr_track.duration);

    seek_slider.value = seekPosition;

    let currentMinutes = Math.floor(curr_track.currentTime / 60);
    let currentSeconds = Math.floor(curr_track.currentTime - currentMinutes * 60);
    let durationMinutes = Math.floor(curr_track.duration / 60);
    let durationSeconds = Math.floor(curr_track.duration - durationMinutes * 60);

    if (currentSeconds < 10) { currentSeconds = "0" + currentSeconds; }
    if (durationSeconds < 10) { durationSeconds = "0" + durationSeconds; }
    if (currentMinutes < 10) { currentMinutes = "0" + currentMinutes; }
    if (durationMinutes < 10) { durationMinutes = "0" + durationMinutes; }

    curr_time.textContent = currentMinutes + ":" + currentSeconds;
    total_duration.textContent = durationMinutes + ":" + durationSeconds;
  }
}

function playpauseTrack() {
  if (!isPlaying) {
    // Load the first track in the tracklist
    loadTrack(slot_cnt);
    playTrack();
    playpause_btn.style.display = 'none';
    let hour = 1, min = 20, sec = 0;
    let totalTimeSec = 75*60;
    let timeLeftSec = 75*60;
    
    timerStart = setInterval(function(){
      timeLeftSec --;
      if (timeLeftSec > 0){
        $('.progress-bar').css('width', timeLeftSec/totalTimeSec*100+'%');
        hour = Math.floor(timeLeftSec/60/60);
        min = Math.floor((timeLeftSec/60)%60);
        sec = timeLeftSec%60;
        $('.countDown').text("Time Left " + String(hour).padStart(1, '0') + ":" + String(min).padStart(2, '0') + ":"+ String(sec).padStart(2, '0'));

      } else {
        clearInterval(timerStart);
        alert("Time's up! You failed to complete the experiment on time!");
        window.onbeforeunload = null;
        window.location.href='index.html';
      }
      
    }, 1000);
  }
}

function progress(timeleft, timetotal, $element) {
  // countdown progress bar (timer) for the experiment 
  let progressBarWidth = timeleft * $element.width() / timetotal;
  
  $('.bar').animate({ width: progressBarWidth }, 500);
  $('.countDown').text(String(hour).padStart(2, '0') + ":" + String(min).padStart(2, '0') + ":"+ String(sec).padStart(2, '0'));
  if(timeleft > 0) {
      setTimeout(function() {
          timeleft = timeleft- 1
          progress();
      }, 1000);
  } else {
    alert("Time's up! You failed to complete the experiment on time!");
    window.onbeforeunload = null;
    window.location.href='index.html';
  }
};


function submitEmail() {
  // close modal if user filled necessary info
  email = document.getElementById("userEmail").value;
  agreed = document.getElementById("agreeCheck").checked
  if (email != "" && agreed == true) {
    console.log('log in as', email);
    submitModal.toggle();
    startTime = new Date().toLocaleString('en-us', {timeZone: 'Asia/Taipei', hour12: false});
    createRowInDB();
  } 
}

function createRowInDB() {
  // create new entry in database for user
  let audioOrderStr = audioOrder.join();
  let responseStr = userResponses.join();
  let responsePositionStr = userResponsePositions.join();
  // https://stackoverflow.com/questions/7820683/convert-boolean-result-into-number-integer
  data = {"startTime": startTime, "email": email, "audioOrderStr": audioOrderStr, "responseStr": responseStr, "responsePositionStr": responsePositionStr, "experimentFinished": +experimentFinished};
  console.log(data);
  $.post('insert_row_to_db.php', data, function (response) {
    console.log(response);
    console.log("create new user data for " + email + " in db !!");
  });
}

function updateDB() {
  // update user data in database
  let audioOrderStr = audioOrder.join(); // save list to db as strings
  let responseStr = userResponses.join();
  let responsePositionStr = userResponsePositions.join();
  let nowTime = new Date().toLocaleString('en-us', {timeZone: 'Asia/Taipei', hour12: false});
  // https://stackoverflow.com/questions/7820683/convert-boolean-result-into-number-integer
  data = {"startTime": startTime, "nowTime": nowTime, "email": email, "audioOrderStr": audioOrderStr, "responseStr": responseStr,
   "responsePositionStr": responsePositionStr, "vigilanceDetail": vigilanceDetail, "vigilanceScore": vigilanceScore, "token": finished_token,
    "experimentFinished": +experimentFinished};

  console.log(data);
  $.post('update_db.php', data, function (response) {
    console.log(response);
    console.log("successfully updated user data to db !!");
  });
}


function copyToClipboard() {
    window.prompt("Copy to clipboard: Ctrl+C, Enter", finished_token);
  }

// popup warning about leaving experiment
window.onbeforeunload = function(){
  return 'Experiment data will be lost';
};

// record the music position in second when user makes decision
$('input[name="toggle"]').on("click", function() {
  userResponseInSec = curr_track.currentTime;
})

// ================ setup before experiment begins ===============
audioOrder = shuffle(audioOrder);

// order of every audio and break (-1 for 3s break, -2 for 3m break)
let slotWithBreakOrder = Array(slotOrder.length*2-1).fill(-1);
for (let i = 0; i < slotWithBreakOrder.length; i++) {
  if (i%2 === 0) {
      // insert break between audios
      slotWithBreakOrder[i] = slotOrder[i/2];
  }    
}
slotWithBreakOrder[slotOrder.length/3*2-1] = -2; //first rest
slotWithBreakOrder[slotOrder.length/3*2*2-1] = -2; //second rest

$('label[for="toggle-hrd"]').hide();
$('label[for="toggle-unhrd"]').hide();
$('label[for="toggle-alhrd"]').hide();
$('.skip_btn').hide();