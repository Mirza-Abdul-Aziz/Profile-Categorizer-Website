var root = document.querySelector(':root')
function light_theme() {
  root.style.setProperty("--body-color", "#d6d6d6");
  root.style.setProperty("--navbar-color", "lightgray");
}
function dark_theme() {
  root.style.setProperty("--body-color", "black");
  root.style.setProperty("--navbar-color", "black");
}