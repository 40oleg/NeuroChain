class ItemD<T, Q> {
    f: T;
    s: Q;
    constructor(_f: T, _s: Q) {
        this.f = _f;
        this.s = _s;
    }
}

export class Dictionary<T, Q> {
    public items: Array<ItemD<T,Q>>;
    constructor() {
        this.items = new Array<ItemD<T,Q>>();
    }
    public Add(key: T, value: Q) {
        this.items.push(new ItemD(key, value))
    }
    Key(key: T): Q {
        for(let i = 0; i < this.items.length; i++) {
            if(this.items[i].f == key) {
                return this.items[i].s;
            }
        }
        throw new Error("Error in method \"Key\"");
    }
    Value(value: Q): T {
        for(let i = 0; i < this.items.length; i++) {
            if(this.items[i].s == value) {
                return this.items[i].f;
            }
        }
        throw new Error("Error in method \"Value\"");
    }
    GetIndex(key: T): number {
        for(let i = 0; i < this.items.length; i++) {
            if(this.items[i].f == key) {
                return i;
            }
        }
        return -1;
    }
}