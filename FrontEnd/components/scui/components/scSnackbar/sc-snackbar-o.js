"use strict";

var SnackBar = function () {
  var t = function (t) {
    if (this.id = t || null, this.id) {
      var n = getCurrentPages(),
          r = n[n.length - 1];
      if (this.snackBar = r.selectComponent(this.id), this.snackBar) return this;
      throw new Error("no this id of sc-snackbar");
    }
  };

  return t.prototype.open = function (t) {
    return this.snackBar._open(t), this;
  }, t.prototype.close = function () {
    return this.snackBar._close(), this;
  }, function (n) {
    return new t(n);
  };
}();

module.exports = SnackBar;