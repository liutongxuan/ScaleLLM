// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             v4.23.4
// source: completion.proto

package scalellm

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

const (
	Completion_Complete_FullMethodName = "/llm.Completion/Complete"
	Completion_Chat_FullMethodName     = "/llm.Completion/Chat"
)

// CompletionClient is the client API for Completion service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type CompletionClient interface {
	// legacy API
	Complete(ctx context.Context, in *CompletionRequest, opts ...grpc.CallOption) (Completion_CompleteClient, error)
	Chat(ctx context.Context, in *ChatRequest, opts ...grpc.CallOption) (Completion_ChatClient, error)
}

type completionClient struct {
	cc grpc.ClientConnInterface
}

func NewCompletionClient(cc grpc.ClientConnInterface) CompletionClient {
	return &completionClient{cc}
}

func (c *completionClient) Complete(ctx context.Context, in *CompletionRequest, opts ...grpc.CallOption) (Completion_CompleteClient, error) {
	stream, err := c.cc.NewStream(ctx, &Completion_ServiceDesc.Streams[0], Completion_Complete_FullMethodName, opts...)
	if err != nil {
		return nil, err
	}
	x := &completionCompleteClient{stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type Completion_CompleteClient interface {
	Recv() (*CompletionResponse, error)
	grpc.ClientStream
}

type completionCompleteClient struct {
	grpc.ClientStream
}

func (x *completionCompleteClient) Recv() (*CompletionResponse, error) {
	m := new(CompletionResponse)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

func (c *completionClient) Chat(ctx context.Context, in *ChatRequest, opts ...grpc.CallOption) (Completion_ChatClient, error) {
	stream, err := c.cc.NewStream(ctx, &Completion_ServiceDesc.Streams[1], Completion_Chat_FullMethodName, opts...)
	if err != nil {
		return nil, err
	}
	x := &completionChatClient{stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type Completion_ChatClient interface {
	Recv() (*ChatResponse, error)
	grpc.ClientStream
}

type completionChatClient struct {
	grpc.ClientStream
}

func (x *completionChatClient) Recv() (*ChatResponse, error) {
	m := new(ChatResponse)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

// CompletionServer is the server API for Completion service.
// All implementations must embed UnimplementedCompletionServer
// for forward compatibility
type CompletionServer interface {
	// legacy API
	Complete(*CompletionRequest, Completion_CompleteServer) error
	Chat(*ChatRequest, Completion_ChatServer) error
	mustEmbedUnimplementedCompletionServer()
}

// UnimplementedCompletionServer must be embedded to have forward compatible implementations.
type UnimplementedCompletionServer struct {
}

func (UnimplementedCompletionServer) Complete(*CompletionRequest, Completion_CompleteServer) error {
	return status.Errorf(codes.Unimplemented, "method Complete not implemented")
}
func (UnimplementedCompletionServer) Chat(*ChatRequest, Completion_ChatServer) error {
	return status.Errorf(codes.Unimplemented, "method Chat not implemented")
}
func (UnimplementedCompletionServer) mustEmbedUnimplementedCompletionServer() {}

// UnsafeCompletionServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to CompletionServer will
// result in compilation errors.
type UnsafeCompletionServer interface {
	mustEmbedUnimplementedCompletionServer()
}

func RegisterCompletionServer(s grpc.ServiceRegistrar, srv CompletionServer) {
	s.RegisterService(&Completion_ServiceDesc, srv)
}

func _Completion_Complete_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(CompletionRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(CompletionServer).Complete(m, &completionCompleteServer{stream})
}

type Completion_CompleteServer interface {
	Send(*CompletionResponse) error
	grpc.ServerStream
}

type completionCompleteServer struct {
	grpc.ServerStream
}

func (x *completionCompleteServer) Send(m *CompletionResponse) error {
	return x.ServerStream.SendMsg(m)
}

func _Completion_Chat_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ChatRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(CompletionServer).Chat(m, &completionChatServer{stream})
}

type Completion_ChatServer interface {
	Send(*ChatResponse) error
	grpc.ServerStream
}

type completionChatServer struct {
	grpc.ServerStream
}

func (x *completionChatServer) Send(m *ChatResponse) error {
	return x.ServerStream.SendMsg(m)
}

// Completion_ServiceDesc is the grpc.ServiceDesc for Completion service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Completion_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "llm.Completion",
	HandlerType: (*CompletionServer)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Complete",
			Handler:       _Completion_Complete_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "Chat",
			Handler:       _Completion_Chat_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "completion.proto",
}