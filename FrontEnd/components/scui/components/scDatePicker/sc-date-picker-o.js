"use strict";

var DatePicker = function () {
  var t = function (t) {
    if (this.id = t || null, this.id) {
      var e = getCurrentPages(),
          i = e[e.length - 1];
      if (this.datePicker = i.selectComponent(this.id), this.datePicker) return this;
      throw new Error("no this id of sc-date-picker");
    }
  };

  return t.prototype.open = function (t) {
    return this.datePicker._open(t), this;
  }, t.prototype.close = function () {
    return this.datePicker._close(), this;
  }, function (e) {
    return new t(e);
  };
}();

module.exports = DatePicker;