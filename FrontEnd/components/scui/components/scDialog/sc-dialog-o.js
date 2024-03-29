"use strict";

var Dialog = function () {
  var t = function (t) {
    if (this.id = t || null, this.id) {
      var o = getCurrentPages(),
          i = o[o.length - 1];
      if (this.dialog = i.selectComponent(this.id), this.dialog) return this;
      throw new Error("no this id of sc-dialog");
    }
  };

  return t.prototype.open = function () {
    return this.dialog._open(), this;
  }, t.prototype.close = function () {
    return this.dialog._close(), this;
  }, t.prototype.toggle = function () {
    return this.dialog.data.opened ? this.close() : this.open();
  }, function (o) {
    return new t(o);
  };
}();

module.exports = Dialog;