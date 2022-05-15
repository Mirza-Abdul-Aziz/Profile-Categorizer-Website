var root = document.querySelector(":root");
// var twitter_logo = document.getElementsByClassName('body');
function light_theme() {
  root.style.setProperty("--body-color", "#d6d6d6");
  root.style.setProperty("--navbar-color", "lightgray");
  //   twitter_logo.style.setProperty("")
}
function dark_theme() {
  root.style.setProperty("--body-color", "black");
  root.style.setProperty("--navbar-color", "black");
}
