"use strict";

var transHexOtTgb = require("../../assets/lib/transHexOrRgb/transHexOrRgb");

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
    label: {
      type: String
    },
    value: {
      type: null
    },
    color: {
      type: String,
      value: "#ff4081"
    },
    reverse: {
      type: Boolean,
      value: !1
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
    showRipple: !1,
    disabled: !1,
    value: null
  },
  ready: function () {
    this.setData({
      checked: this.properties.checked,
      disabled: this.properties.disabled,
      value: this.properties.value
    });
  },
  externalClasses: ["sc-class"],
  methods: {
    _changeRadio: function () {
      this.setData({
        checked: !0,
        clicked: !0,
        showRipple: !0
      }), this.triggerEvent("radiochange", {
        value: this.properties.value
      }, {
        bubbles: !0,
        composed: !0
      });
    },
    _animationend: function () {
      this.setData({
        showRipple: !1
      });
    }
  }
});