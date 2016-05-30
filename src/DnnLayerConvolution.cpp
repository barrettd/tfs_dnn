// --------------------------------------------------------------------
//  DnnLayerConvolution.cpp
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#include "DnnLayerConvolution.h"
#include "Error.h"

namespace tfs {
  
    static const char *NAME = "conv";

    const char*
    DnnLayerConvolution::className( void ) {
        return NAME;
    }

    DnnLayerConvolution::DnnLayerConvolution( DnnLayer *previousLayer,
                                             unsigned long side,
                                             unsigned long filters,
                                             unsigned long stride,
                                             unsigned long pad,
                                             const bool trainable ):
    DnnLayer( NAME, previousLayer ),
    m_side(    side ),
    m_filters( filters ),
    m_stride(  stride ),
    m_pad(     pad ) {              // Constructor
        m_l1_decay_mul = 0.0;
        m_l2_decay_mul = 1.0;
        if( previousLayer != 0 ) {  // previousLayer should not be null.
            setup( trainable );
        } else {
            log_error( "previousLayer is null" );
        }
    }
    
    DnnLayerConvolution::~DnnLayerConvolution( void ) {
        // Destructor
    }
    
    void
    DnnLayerConvolution::setup( const bool trainable ) {
        // -----------------------------------------------------------------------------------
        // N = number of filters
        // S = size of input data
        //  w[N,S+1] = filters weight + bias weight
        // dw[N,S+1] = gradiant + d/dw bias
        // out_a[N]  = activations
        // out_dw[N] = gradiant
        // -----------------------------------------------------------------------------------
        if( m_side < 1 || m_filters < 1 || m_stride < 1 ) {
            log_error( "Bad params" );
            return;
        }
        if( m_in_a == 0 ) {
            log_error( "Input activation matrix is null" );
            return;
        }
        const unsigned long in_x = m_in_a->width();
        const unsigned long in_y = m_in_a->height();
        const unsigned long in_z = m_in_a->depth();
        
        const unsigned long out_x = floorl((in_x + m_pad * 2 - m_side) / m_stride + 1 );
        const unsigned long out_y = floorl((in_y + m_pad * 2 - m_side) / m_stride + 1 );
        const unsigned long out_z = m_filters;
        
        const unsigned long N = m_filters;
        const unsigned long S = m_side * m_side * in_z;     // 1d input, S elements.
        
        m_w = new Matrix( N, S+1, 1 );                      // weights N x (S+1)
        if( trainable ) {
            m_dw = new Matrix( N, S+1, 1 );                 // gradiant N x (S+1)
        }
        m_out_a = new Matrix( out_x, out_y, out_z );        // Activations (output)
        if( trainable ) {
            m_out_dw = new Matrix( out_x, out_y, out_z );   // Activation gradiant
        }

        return;
    }
    
    bool
    DnnLayerConvolution::runForward(  void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Forward propagate, used with forward()
        // -----------------------------------------------------------------------------------
        // TODO:
        
//        this.in_act = V;
//        var A = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);
//        
//        var V_sx = V.sx |0;
//        var V_sy = V.sy |0;
//        var xy_stride = this.stride |0;
//        
//        for(var d=0;d<this.out_depth;d++) {
//            var f = this.filters[d];
//            var x = -this.pad |0;
//            var y = -this.pad |0;
//            for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
//                x = -this.pad |0;
//                for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride
//                    
//                    // convolve centered at this particular location
//                    var a = 0.0;
//                    for(var fy=0;fy<f.sy;fy++) {
//                        var oy = y+fy; // coordinates in the original input array coordinates
//                        for(var fx=0;fx<f.sx;fx++) {
//                            var ox = x+fx;
//                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
//                                for(var fd=0;fd<f.depth;fd++) {
//                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
//                                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
//                                }
//                            }
//                        }
//                    }
//                    a += this.biases.w[d];
//                    A.set(ax, ay, d, a);
//                }
//            }
//        }
//        this.out_act = A;
//        return this.out_act;
        return true;
    }
    
    bool
    DnnLayerConvolution::runBackprop( void ) {
        // -----------------------------------------------------------------------------------
        // virtual: Back propagate, used with backprop()
        // -----------------------------------------------------------------------------------
        // TODO:
//    backward: function() {
//        
//        var V = this.in_act;
//        V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it
//        
//        var V_sx = V.sx |0;
//        var V_sy = V.sy |0;
//        var xy_stride = this.stride |0;
//        
//        for(var d=0;d<this.out_depth;d++) {
//            var f = this.filters[d];
//            var x = -this.pad |0;
//            var y = -this.pad |0;
//            for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
//                x = -this.pad |0;
//                for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride
//                    
//                    // convolve centered at this particular location
//                    var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
//                    for(var fy=0;fy<f.sy;fy++) {
//                        var oy = y+fy; // coordinates in the original input array coordinates
//                        for(var fx=0;fx<f.sx;fx++) {
//                            var ox = x+fx;
//                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
//                                for(var fd=0;fd<f.depth;fd++) {
//                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
//                                    var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
//                                    var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
//                                    f.dw[ix2] += V.w[ix1]*chain_grad;
//                                    V.dw[ix1] += f.w[ix2]*chain_grad;
//                                }
//                            }
//                        }
//                    }
//                    this.biases.dw[d] += chain_grad;
//                }
//            }
//        }
        return true;
    }

    
    
}   // namespace tfs