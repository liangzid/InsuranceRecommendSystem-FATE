"use strict";

var scRippleBehaviors = require("../sc-ripple-behaviors/sc-ripple-behaviors");

Component({
  options: {
    multipleSlots: !0
  },
  behaviors: [scRippleBehaviors],
  properties: {
    subHeader: {
      type: String
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  externalClasses: ["sc-class", "left-icon", "right-icon"],
  methods: {
    _tap: function (e) {
      this._addRipple_(e);
    },
    _longPress: function (e) {
      this._longPress_(e);
    },
    _rippleAnimationEnd: function () {
      this._rippleAnimationEnd_();
    },
    _touchEnd: function () {
      this._touchEnd_();
    }
  }
});