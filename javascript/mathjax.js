window.MathJax = {
  tex: {
    inlineMath: [ ['$', '$'], ["\\(", "\\)"]],
    displayMath: [ ['$$','$$'], ["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})