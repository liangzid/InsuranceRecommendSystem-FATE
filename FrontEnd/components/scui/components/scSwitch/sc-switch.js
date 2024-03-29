"use strict";

var swicthCount = 1;
Component({
  properties: {
    checked: {
      type: Boolean,
      value: !1
    },
    disabled: {
      type: Boolean,
      value: !1
    },
    name: {
      type: String
    },
    color: {
      type: String,
      value: "#ff4081"
    },
    ripple: {
      type: Boolean,
      value: !0
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    checked: !1,
    clicked: !1,
    value: null
  },
  ready: function () {
    var e = this.properties,
        t = e.checked,
        a = e.name,
        c = void 0 === a ? "switch" + swicthCount++ : a;
    this.setData({
      checked: t,
      value: t,
      name: c
    });
  },
  externalClasses: ["sc-class"],
  methods: {
    _changeSwitch: function () {
      var e = this.data,
          t = e.checked,
          a = e.name;
      t = !t, this.setData({
        checked: t,
        clicked: !0,
        value: t
      }), this.triggerEvent("change", {
        name: a,
        value: t
      }, {
        bubbles: !0
      });
    },
    _animationend: function () {
      this.setData({
        clicked: !1
      });
    }
  }
});