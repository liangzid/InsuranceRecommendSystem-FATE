"use strict";

Component({
  properties: {
    ripple: {
      type: Boolean,
      value: !0
    },
    tabList: {
      type: Array
    },
    tabIndex: {
      type: Number,
      value: 0
    },
    transition: {
      type: Boolean,
      value: !0
    },
    driverColor: {
      type: String,
      value: "#0088CC"
    },
    swanIdForSystem: {
      type: String,
      value: "123445"
    }
  },
  data: {
    currentTab: 0,
    touchMoveStartX: 0,
    touchMoveEndX: 0,
    btnMinWidth: 88,
    scrollLeft: 0,
    scroll: 0,
    scrollViewWidth: 0,
    moveLength: 64,
    allTabItem: [],
    tabDriverWidth: 0,
    tabDriverLeft: 0,
    isUnEqualNextWidth: !1
  },
  externalClasses: ["sc-class", "active-class"],
  ready: function () {
    var t = this;
    this._queryMultipleNodes("#sc-tab").then(function (a) {
      t.data.scrollViewWidth = a[0].width;
    }), this._queryAllNodes(".tab-bar-item").then(function (a) {
      var e = t.data.allTabItem = a;
      t.data.tabStartPosition = e[0].left, t.data.tabEndPosition = e[e.length - 1].right, t.data.tabListWinth = t.data.tabEndPosition - t.data.tabStartPosition, t.data.tabs = t._getAllTab(), t._setTab(t.properties.tabIndex || 0);
    });
  },
  methods: {
    _queryMultipleNodes: function (t) {
      var a = this;
      return new Promise(function (e, i) {
        var s = a.createSelectorQuery();
        s.select(t).boundingClientRect(), s.selectViewport().scrollOffset(), s.exec(function (t) {
          e(t);
        });
      });
    },
    _queryAllNodes: function (t) {
      var a = this;
      return new Promise(function (e, i) {
        var s = a.createSelectorQuery();
        s.selectAll(t).boundingClientRect(), s.exec(function (t) {
          e(t[0]);
        });
      });
    },
    _tabBarTap: function (t) {
      var a = t.target.dataset.idx;

      if (this.data.currentTab !== a) {
        var e = this.data.allTabItem[this.data.currentTab].width,
            i = this.data.allTabItem[a].width;
        this.setData({
          isUnEqualNextWidth: e !== i,
          currentTab: a
        }), this._setTab(a);
      }
    },
    _setTab: function (t) {
      if (this.data.scrollViewWidth < this.data.tabListWinth) this._setScroll(t);else {
        var a = this.data.allTabItem[t],
            e = a.left,
            i = a.width;
        this.setData({
          tabDriverWidth: i,
          tabDriverLeft: e - this.data.tabStartPosition
        });
      }
      this._setHeight(t), this.triggerEvent("change", {
        value: t,
        data: this.data.tabList[t].data || {}
      });
    },
    _setScroll: function (t) {
      var a = this.data,
          e = a.scrollLeft,
          i = a.scrollViewWidth,
          s = a.tabStartPosition,
          r = this.data.allTabItem[t],
          l = r.left,
          h = r.right,
          n = r.width;
      this.setData({
        tabDriverWidth: n,
        tabDriverLeft: l - s
      }), l >= e + s && h <= Math.ceil(e + i + s) || (l - s <= e ? this.setData({
        scrollLeft: 0 === t ? 0 : l - s - (t > 0 ? this.data.allTabItem[t - 1].width / 2 : 0)
      }) : this.setData({
        scrollLeft: h - i - s + (t + 1 < this.data.tabList.length ? this.data.allTabItem[t + 1].width / 2 : 0)
      }));
    },
    _setHeight: function (t) {
      var a = this.data.tabs;
      setTimeout(function () {
        for (var e = 0; e < a.length; e++) e === t ? a[e].setData({
          tabHeight: "auto"
        }) : a[e].setData({
          tabHeight: 0
        });
      }, 150);
    },
    _touchStart: function (t) {
      this.setData({
        touchMoveStartX: t.touches[0].clientX
      });
    },
    _touchEnd: function (t) {
      if (1 === t.changedTouches.length) {
        var a = this.data,
            e = a.currentTab,
            i = a.moveLength,
            s = a.scrollViewWidth,
            r = a.btnMinWidth,
            l = this.properties.tabList.length,
            h = t.changedTouches[0].clientX - this.data.touchMoveStartX;

        if (this.setData({
          touchMoveEndX: 0
        }), Math.abs(h) > i) {
          h >= 0 ? this.setData({
            currentTab: e - 1 >= 0 ? e - 1 : 0
          }) : this.setData({
            currentTab: e + 1 < l ? e + 1 : e
          });
          var n = this.data.currentTab;

          if (e !== n) {
            var o = this.data.allTabItem[e].width,
                c = this.data.allTabItem[n],
                d = c.left,
                b = c.width;
            this.setData({
              isUnEqualNextWidth: o !== b
            }), s >= l * r ? this.setData({
              tabDriverWidth: b,
              tabDriverLeft: d - this.data.tabStartPosition
            }) : this._setScroll(n), this._setHeight(n), this.triggerEvent("change", {
              value: n,
              data: this.data.tabList[n].data || {}
            });
          }
        }
      }
    },
    _scroll: function (t) {
      this.data.scrollLeft = t.detail.scrollLeft;
    },
    _getAllTab: function () {
      return getCurrentPages()[getCurrentPages().length - 1].selectAllComponents('.scTab-' + this.data.swanIdForSystem);
    }
  }
});